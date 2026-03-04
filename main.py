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
    "--csv", type=str, help="Path to CSV with SONG_ID, SONG_TITLE, ARTIST columns"
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
    "SPECTROGRAM_PATH_MEL",
    "SPECTROGRAM_PATH_LINEAR",
    "SPECTROGRAM_MATRIX_MEL",  # raw numpy matrix (128 x T), base64-encoded .npy
    "SPECTROGRAM_B64_LINEAR",  # PNG base64 kept for human reference
    "BOUNDARIES_1",
    "SEGMENTS_JSON_1",
    "BOUNDARIES_2",
    "SEGMENTS_JSON_2",
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
                "SPECTROGRAM_PATH_MEL": mel_path,
                "SPECTROGRAM_PATH_LINEAR": linear_path,
                "SPECTROGRAM_MATRIX_MEL": matrix_to_base64(mel_matrix),
                "SPECTROGRAM_B64_LINEAR": img_to_base64(linear_path),
                "BOUNDARIES_1": json.dumps(boundaries1),
                "SEGMENTS_JSON_1": json.dumps(segments1),
                "BOUNDARIES_2": json.dumps(boundaries2),
                "SEGMENTS_JSON_2": json.dumps(segments2),
            }
        )


def parse_textfile(path):
    """
    Parse a SALAMI annotation text file.
    Returns:
      - segments: list of {t, raw_label, major, function} dicts (full JSON data)
      - boundaries: list of timestamps where significant changes occur
                    (uppercase letter change OR named function like Verse/Bridge/Chorus/Intro/Outro)
    """
    NAMED_FUNCTIONS = {
        "verse",
        "chorus",
        "bridge",
        "intro",
        "outro",
        "prechorus",
        "pre-chorus",
        "transition",
        "interlude",
        "solo",
        "coda",
    }

    segments = []
    boundaries = []
    prev_major = None
    prev_functions = set()

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t", 1)
                if len(parts) < 2:
                    continue

                t = float(parts[0])
                raw_label = parts[1].strip()

                components = [
                    c.strip().rstrip(",") for c in re.split(r",\s*", raw_label)
                ]
                major = next((c for c in components if re.match(r"^[A-Z]$", c)), None)
                functions = {
                    c.lower().strip("()")
                    for c in components
                    if c.lower().strip("()") in NAMED_FUNCTIONS
                }

                segments.append(
                    {
                        "t": t,
                        "raw_label": raw_label,
                        "major": major,
                        "functions": list(functions),
                    }
                )

                is_boundary = False
                if major and major != prev_major:
                    is_boundary = True
                if functions and not functions.issubset(prev_functions):
                    is_boundary = True
                if raw_label.lower() in ("silence", "end", "applause", "noise"):
                    is_boundary = True

                if is_boundary:
                    boundaries.append(t)

                if major:
                    prev_major = major
                prev_functions = functions

    except FileNotFoundError:
        print(f"  ⚠ Annotation file not found: {path}")
    except Exception as e:
        print(f"  ⚠ Could not parse {path}: {e}")

    return segments, boundaries


# ── Load song list ─────────────────────────────────────────────────────────────

entries = []

if args.txt:
    if not os.path.exists(args.txt):
        print(f"ERROR: {args.txt} not found.")
        sys.exit(1)
    with open(args.txt) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                parts = line.split(" - ", 1)
                artist = parts[0].strip() if len(parts) == 2 else "Unknown"
                title = parts[1].strip() if len(parts) == 2 else line
                entries.append(
                    {
                        "song_id": str(i + 1),
                        "song_title": title,
                        "artist": artist,
                        "query": line,
                    }
                )

elif args.csv:
    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found.")
        sys.exit(1)
    df = pd.read_csv(args.csv)
    for _, row in df.iterrows():
        title = clean(str(row["SONG_TITLE"]))
        artist = clean(str(row["ARTIST"]))
        song_id = str(row["SONG_ID"])
        annotation_dir = os.path.join("annotations", song_id)
        textfile1 = os.path.join(annotation_dir, "textfile1.txt")
        textfile2 = os.path.join(annotation_dir, "textfile2.txt")
        if title and artist:
            entries.append(
                {
                    "song_id": song_id,
                    "song_title": title,
                    "artist": artist,
                    "query": f"{artist} - {title}",
                    "textfile1": textfile1,
                    "textfile2": textfile2,
                }
            )

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
            info = ydl.extract_info(query, download=True)
            if "entries" in info:
                if not info["entries"]:
                    raise ValueError("No search results found")
                yt_title = info["entries"][0]["title"]
            else:
                yt_title = info["title"]
            audio_path = os.path.join(AUDIO_DIR, f"{yt_title}.wav")

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
        textfile1 = entry.get("textfile1", "")
        textfile2 = entry.get("textfile2", "")
        segments1, boundaries1 = parse_textfile(textfile1) if textfile1 else ([], [])
        segments2, boundaries2 = parse_textfile(textfile2) if textfile2 else ([], [])
        if boundaries1:
            print(f"  ✓ Parsed {len(boundaries1)} boundaries from annotator 1")
        if boundaries2:
            print(f"  ✓ Parsed {len(boundaries2)} boundaries from annotator 2")

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

# ── Build merged dataset ───────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"Done! {len(success)}/{len(entries)} succeeded.")

if args.csv and os.path.exists(MANIFEST_FILE):
    print(f"\nJoining manifest back to original dataset...")
    original_df = pd.read_csv(args.csv)
    manifest_df = pd.read_csv(MANIFEST_FILE)

    # Normalise SONG_ID to string for safe join
    original_df["SONG_ID"] = original_df["SONG_ID"].astype(str)
    manifest_df["SONG_ID"] = manifest_df["SONG_ID"].astype(str)

    # Drop any spectrogram columns that already exist in original_df to avoid duplicates
    cols_to_drop = [
        c for c in MANIFEST_MERGE_COLS if c != "SONG_ID" and c in original_df.columns
    ]
    if cols_to_drop:
        print(
            f"  Dropping stale columns from original to avoid duplicates: {cols_to_drop}"
        )
        original_df = original_df.drop(columns=cols_to_drop)

    # Deduplicate manifest (safety net if a song was written twice)
    manifest_slim = manifest_df[MANIFEST_MERGE_COLS].drop_duplicates(
        subset=["SONG_ID"], keep="last"
    )

    merged_df = original_df.merge(manifest_slim, on="SONG_ID", how="inner")
    merged_df.to_csv(MERGED_FILE, index=False)
    print(f"✓ Merged dataset saved to {MERGED_FILE} ({len(merged_df)} rows)")

elif not args.csv:
    print(f"(No original CSV to join against — manifest saved to {MANIFEST_FILE})")

if failed:
    print(f"\nFailed ({len(failed)}):")
    for e in failed:
        print(f"  - {e['artist']} - {e['song_title']}  (ID: {e['song_id']})")
    with open("failed_songs.txt", "w") as f:
        f.write("\n".join([f"{e['artist']} - {e['song_title']}" for e in failed]))
    print("\nFailed songs saved to failed_songs.txt")

# ── Training-time loading snippet ─────────────────────────────────────────────
# import base64, io, numpy as np
# buf = io.BytesIO(base64.b64decode(row["SPECTROGRAM_MATRIX_MEL"]))
# matrix = np.load(buf)   # shape: (128, T) — 128 mel bands x time frames
