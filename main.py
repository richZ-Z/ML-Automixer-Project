import yt_dlp
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time
import sys
import random
import argparse
import base64
import json
import re
import io
import csv
import difflib

# ── Config ────────────────────────────────────────────────────────────────────

AUDIO_DIR = "audio"  # temp folder for downloaded audio
SPECTROGRAM_DIR = "spectrograms/mel"  # output folder for mel spectrogram PNGs
LINEAR_DIR = (
    "spectrograms/linear"  # output folder for linear spectrogram PNGs (human reference)
)
MANIFEST_FILE = "manifest.csv"  # built as we go
MERGED_FILE = "dataset_with_spectrograms.csv"  # original dataset joined with manifest

N_MELS = 128  # mel bands (64=small, 128=standard, 256=detailed)
FMAX = 8000  # max frequency in Hz
IMG_SIZE = (4, 4)  # output image size in inches
SLEEP_BETWEEN = 3  # seconds between downloads

# True = clean image for CNN (no axes, no padding)
# False = labeled image for human viewing
ML_MODE = True

# ── Argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Download audio and generate mel spectrograms."
)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument(
    "--txt", type=str, help="Path to txt file (one 'Artist - Song' per line)"
)
group.add_argument(
    # for now, we expect the SALAMI iTunes library CSV to be pre-downloaded and cleaned manually into this file
    # might add ability to read different csv files in the future, but this is just a one-off script for now
    "--csv", type=str,
    help=(
        "Path to one or more CSV files (comma-separated) or a directory "
        "containing SALAMI metadata.  Files must include SONG_ID (or salami_id), "
        "a title field (TITLE / TITLE_IN_SALAMI / SONG_TITLE / Name), and ARTIST."
    )
    # can type any string here since we hardcode the filename 
)
parser.add_argument(
    "--n", type=int, default=None, help="Number of songs to process (default: all)"
)
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed for shuffle (default: 42)"
)
parser.add_argument("--no-shuffle", action="store_true", help="Disable shuffling")
args = parser.parse_args()

# ── Helpers ───────────────────────────────────────────────────────────────────


def clean(s):
    """Replace underscores with spaces and strip whitespace."""
    return s.replace("_", " ").strip()


def matrix_to_base64(matrix: np.ndarray) -> str:
    """
    Encode a numpy array (e.g. mel spectrogram matrix) as a compressed base64 string.
    Stores as .npy format inside a BytesIO buffer — no lossy image encoding.
    """
    buf = io.BytesIO()
    np.save(buf, matrix)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def base64_to_matrix(b64_str: str) -> np.ndarray:
    """
    Decode a base64 string back to a numpy array.
    Usage at training time:
        matrix = base64_to_matrix(row["SPECTROGRAM_MATRIX_MEL"])
        # shape: (128, T) — 128 mel bands x time frames
    """
    buf = io.BytesIO(base64.b64decode(b64_str))
    return np.load(buf)


def img_to_base64(path: str) -> str:
    """Read a PNG file and return it as a base64-encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_processed_ids() -> set:
    """Return a set of SONG_IDs already in the manifest so we can skip them."""
    if not os.path.exists(MANIFEST_FILE):
        return set()
    try:
        df = pd.read_csv(MANIFEST_FILE, usecols=["SONG_ID"])
        return set(df["SONG_ID"].astype(str).tolist())
    except Exception:
        return set()


# Manifest columns — single source of truth so appender and merger stay in sync
MANIFEST_FIELDS = [
    "SONG_ID",
    "SONG_TITLE",
    "ARTIST",
    "BOUNDARIES_1",
    "SEGMENTS_JSON_1",
    "BOUNDARIES_2",
    "SEGMENTS_JSON_2",
    "SPECTROGRAM_PATH_MEL",
    "SPECTROGRAM_PATH_LINEAR",
    "SPECTROGRAM_MATRIX_MEL",  # raw numpy matrix (128 x T), base64-encoded .npy
    "SPECTROGRAM_B64_LINEAR",  # PNG base64 kept for human reference
]

# Columns to carry over into the merged dataset (excludes redundant title/artist)
MANIFEST_MERGE_COLS = [c for c in MANIFEST_FIELDS if c not in ("SONG_TITLE", "ARTIST")]


def append_to_manifest(
    song_id,
    song_title,
    artist,
    mel_path,
    linear_path,
    mel_matrix,
    boundaries1,
    segments1,
    boundaries2,
    segments2,
):
    """Write one row to manifest CSV, creating it with headers if needed."""
    file_exists = os.path.exists(MANIFEST_FILE)
    with open(MANIFEST_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "SONG_ID": song_id,
                "SONG_TITLE": song_title,
                "ARTIST": artist,
                "BOUNDARIES_1": json.dumps(boundaries1),
                "SEGMENTS_JSON_1": json.dumps(segments1),
                "BOUNDARIES_2": json.dumps(boundaries2),
                "SEGMENTS_JSON_2": json.dumps(segments2),
                "SPECTROGRAM_PATH_MEL": mel_path,
                "SPECTROGRAM_PATH_LINEAR": linear_path,
                "SPECTROGRAM_MATRIX_MEL": matrix_to_base64(mel_matrix),
                "SPECTROGRAM_B64_LINEAR": img_to_base64(linear_path),
            }
        )


# helper for already-parsed annotation files

def load_functions_file(path: str):
    """Read *_functions.txt and return (segments, boundaries).

    *segments* is a list of dicts with keys ``t`` and ``f``.
    *boundaries* is just the list of timestamps.

    The files live in ``annotations/<id>/parsed/`` so we look there
    instead of attempting to parse raw SALAMI textfiles.
    """
    segments = []
    boundaries = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t", 1)
                if len(parts) < 2:
                    continue
                try:
                    t = float(parts[0])
                except ValueError:
                    continue
                label = parts[1]
                segments.append({"t": t, "f": label})
                boundaries.append(t)
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return segments, boundaries


# ── Load song list ─────────────────────────────────────────────────────────────

entries = []

if args.txt:
    # not edited by darius lowk not functional rn
    with open(args.txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(" - ", 1)
            if len(parts) != 2:
                continue
            artist = clean(parts[0])
            song_title = clean(parts[1])
            if not artist or not song_title:
                continue
            query = f"ytsearch1:{artist} - {song_title} official audio topic -live -remix -cover"
            entries.append({
                "song_id": f"{artist}_{song_title}".replace(" ", "_"),
                "song_title": song_title,
                "artist": artist,
                "query": query,
                "textfile1": "",
                "textfile2": ""
            })

elif args.csv:

    # Accept either a single CSV, multiple comma-separated CSVs, or a directory
    # containing CSV files.  Concatenate them all so the script can harvest as
    # much SALAMI metadata as possible.
    csv_path = args.csv
    paths = []
    if os.path.isdir(csv_path):
        for fname in sorted(os.listdir(csv_path)):
            if fname.lower().endswith(".csv"):
                paths.append(os.path.join(csv_path, fname))
    else:
        # split on commas in case user provided a list of files
        for part in csv_path.split(","):
            part = part.strip()
            if not part:
                continue
            if not os.path.exists(part):
                print(f"ERROR: CSV file not found: {part}")
                sys.exit(1)
            paths.append(part)

    if not paths:
        print(f"ERROR: no CSV files found in '{csv_path}'")
        sys.exit(1)

    # read and combine only those dataframes that contain at least
    # ARTIST and one of the potential title columns. some of the SALAMI
    # metadata files (codaich, etc.) don't contain any song info and would
    # otherwise break the logic later on.
    df_list = []
    for path in paths:
        tmp = pd.read_csv(path, nrows=0)
        tmp.columns = tmp.columns.str.strip()
        # check for mandatory artist column and at least one title column
        has_artist = "ARTIST" in tmp.columns
        has_title = "TITLE" in tmp.columns
        if not (has_artist and has_title):
            print(f"Skipping {os.path.basename(path)}: missing ARTIST/title columns")
            continue
        # re-read full dataframe now that we've decided to keep it
        full = pd.read_csv(path)
        full.columns = full.columns.str.strip()
        df_list.append(full)
    if not df_list:
        print("ERROR: no suitable CSV files found (must contain ARTIST + title column)")
        sys.exit(1)
    df = pd.concat(df_list, ignore_index=True)

    # title column should be named TITLE in all files
    title_col = "TITLE"
    if title_col not in df.columns:
        print("ERROR: CSV is missing TITLE column")
        sys.exit(1)

    # they all share id
    id_col = "SONG_ID"

    # artist column should be named ARTIST in all files
    artist_col = "ARTIST"
    if artist_col not in df.columns:
        print("ERROR: CSV is missing ARTIST column")
        sys.exit(1)

    # ensure id field is string to maintain compatibility with manifest lookups
    df[id_col] = df[id_col].astype(str)

    # -------------------------
    # Build entries
    # -------------------------
    for _, row in df.iterrows():
        song_id = row[id_col]
        song_title = row.get(title_col)
        artist = row.get(artist_col)

        if pd.isna(song_title) or pd.isna(artist):
            continue

        song_title = clean(str(song_title))
        artist = clean(str(artist))

        if not song_title or not artist:
            continue

        query = f"ytsearch1:{artist} {song_title}"

        # Annotation folder structure (same as before)
        annotation_dir = os.path.join("annotations", song_id, "parsed")
        textfile1 = os.path.join(annotation_dir, "textfile1_functions.txt")
        textfile2 = os.path.join(annotation_dir, "textfile2_functions.txt")

        entries.append({
            "song_id": song_id,
            "song_title": song_title,
            "artist": artist,
            "query": query,
            "textfile1": textfile1,
            "textfile2": textfile2,
        })

if not entries:
    print("ERROR: No songs found in input file.")
    sys.exit(1)

# Shuffle
if not args.no_shuffle:
    random.seed(args.seed)
    random.shuffle(entries)

# Limit to N
if args.n is not None:
    entries = entries[: args.n]

# ── Skip already-processed songs ──────────────────────────────────────────────

processed_ids = load_processed_ids()
if processed_ids:
    before = len(entries)
    entries = [e for e in entries if e["song_id"] not in processed_ids]
    skipped = before - len(entries)
    if skipped:
        print(f"Resuming: skipping {skipped} already-processed song(s).")

if not entries:
    print("All songs already processed — nothing to do.")
    sys.exit(0)

print(
    f"Processing {len(entries)} songs (shuffled={'no' if args.no_shuffle else 'yes'}, seed={args.seed})\n"
)

# ── Setup ─────────────────────────────────────────────────────────────────────

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(SPECTROGRAM_DIR, exist_ok=True)
os.makedirs(LINEAR_DIR, exist_ok=True)

ydl_opts = {
    "format": "bestaudio/best",
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "0",
        }
    ],
    "outtmpl": f"{AUDIO_DIR}/%(title)s.%(ext)s",
    "default_search": "ytsearch1",
    "quiet": True,
    "no_warnings": True,
    "cookiesfrombrowser": ("chrome",),  # swap to ("firefox",) if needed
}

# ── Main loop ─────────────────────────────────────────────────────────────────

success, failed = [], []

for i, entry in enumerate(entries, 1):
    song_id = entry["song_id"]
    song_title = entry["song_title"]
    artist = entry["artist"]
    query = entry["query"]

    print(f"[{i}/{len(entries)}] {artist} - {song_title}  (ID: {song_id})")

    # 1. Download audio
    audio_path = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # First, search for videos without downloading
            print(f"  Searching YouTube for: {query}")
            search_info = ydl.extract_info(query, download=False)
            if "entries" not in search_info or not search_info["entries"]:
                raise ValueError("No search results found")
            entries_list = search_info["entries"]
            # Find the best match using difflib
            best_entry = max(entries_list, key=lambda e: difflib.SequenceMatcher(None, query.lower(), e.get('title', '').lower()).ratio())
            best_title = best_entry['title']
            best_id = best_entry['id']
            similarity = difflib.SequenceMatcher(None, query.lower(), best_title.lower()).ratio()
            print(f"  Best match: {best_title} (similarity: {similarity:.2f})")
            if similarity < 0.3:
                raise ValueError(f"Best match similarity {similarity:.2f} below threshold 0.3")
            # Now download the best match
            ydl.extract_info(f"https://www.youtube.com/watch?v={best_id}", download=True)
            audio_path = os.path.join(AUDIO_DIR, f"{best_title}.wav")

        if not os.path.exists(audio_path):
            wavs = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
            if wavs:
                audio_path = os.path.join(
                    AUDIO_DIR,
                    max(
                        wavs, key=lambda f: os.path.getmtime(os.path.join(AUDIO_DIR, f))
                    ),
                )
            else:
                raise FileNotFoundError("No wav file found after download")

        print(f"  ✓ Downloaded: {os.path.basename(audio_path)}")

    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        failed.append(entry)
        continue

    # 2. Create spectrograms
    safe_name = f"{song_id}_{artist}_{song_title}".replace("/", "-").replace(" ", "_")
    mel_path = os.path.join(SPECTROGRAM_DIR, f"{safe_name}.png")
    linear_path = os.path.join(LINEAR_DIR, f"{safe_name}.png")

    try:
        y, sr = librosa.load(audio_path)

        # ── Mel spectrogram ───────────────────────────────────────────────────
        S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX)
        S_mel_db = librosa.power_to_db(S_mel, ref=np.max)
        # S_mel_db shape: (N_MELS, T) — stored as raw matrix in manifest

        # Also save PNG for visual inspection
        plt.figure(figsize=IMG_SIZE)
        if ML_MODE:
            librosa.display.specshow(
                S_mel_db, sr=sr, x_axis="time", y_axis="mel", fmax=FMAX
            )
            plt.axis("off")
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(mel_path, bbox_inches="tight", pad_inches=0, dpi=150)
        else:
            librosa.display.specshow(
                S_mel_db, sr=sr, x_axis="time", y_axis="mel", fmax=FMAX
            )
            plt.colorbar(format="%+2.0f dB")
            plt.title(f"{artist} - {song_title} (Mel)")
            plt.tight_layout()
            plt.savefig(mel_path, dpi=150)
        plt.close()
        print(f"  ✓ Mel spectrogram saved  (matrix shape: {S_mel_db.shape})")

        # ── Linear spectrogram (for human reference) ─────────────────────────
        S_linear = np.abs(librosa.stft(y))
        S_linear_db = librosa.amplitude_to_db(S_linear, ref=np.max)

        plt.figure(figsize=(12, 4))
        librosa.display.specshow(S_linear_db, sr=sr, x_axis="time", y_axis="hz")
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"{artist} - {song_title}")
        plt.tight_layout()
        plt.savefig(linear_path, dpi=150)
        plt.close()
        print(f"  ✓ Linear spectrogram saved")

        # 3. Parse annotation text files
        # read already-parsed segmentation files
        textfile1 = entry.get("textfile1", "")
        textfile2 = entry.get("textfile2", "")
        segments1, boundaries1 = (
            load_functions_file(textfile1) if textfile1 and os.path.exists(textfile1) else ([], []))
        segments2, boundaries2 = (
            load_functions_file(textfile2) if textfile2 and os.path.exists(textfile2) else ([], []))
        if boundaries1:
            print(f"  ✓ Loaded {len(boundaries1)} boundaries from annotator 1")
        if boundaries2:
            print(f"  ✓ Loaded {len(boundaries2)} boundaries from annotator 2")

        # 4. Write to manifest immediately (progress saved even if we crash mid-run)
        append_to_manifest(
            song_id,
            song_title,
            artist,
            mel_path,
            linear_path,
            S_mel_db,
            boundaries1,
            segments1,
            boundaries2,
            segments2,
        )
        success.append(entry)

    except Exception as e:
        print(f"  ✗ Spectrogram failed: {e}")
        failed.append(entry)

    # 5. Delete audio
    try:
        os.remove(audio_path)
        print(f"  ✓ Audio deleted")
    except Exception as e:
        print(f"  ✗ Could not delete audio: {e}")

    # 6. Rate limiting pause
    if i < len(entries):
        time.sleep(SLEEP_BETWEEN)

# ── Donesies ───────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"Done! {len(success)}/{len(entries)} succeeded.")
