import yt_dlp
import os
from pydub import AudioSegment
import pandas as pd
import json

manifest = pd.read_csv('manifest_wo_classical_consolidated.csv')
print(manifest.shape)

# Look at one song's segments
row = manifest.iloc[1]
print(row['SONG_TITLE'], '-', row['ARTIST'])
print(json.loads(row['SEGMENTS_JSON_1']))


def find_switch_point(segments, label='Chorus', which='last'):
    """Find timestamp of first or last occurrence of a label in segments."""
    matches = [s['t'] for s in segments if s['f'] == label]
    if not matches:
        return None
    return matches[-1] if which == 'last' else matches[0]

def get_segments(row, source=1):
    """Parse segments from manifest row."""
    col = f'SEGMENTS_JSON_{source}'
    val = row.get(col, '')
    if pd.isna(val) or val == '':
        return []
    try:
        return json.loads(val)
    except:
        return []

# Test it on Smashing Pumpkins
segments = get_segments(row, source=1)
switch_out = find_switch_point(segments, label='Chorus', which='last')
switch_in = find_switch_point(segments, label='Verse', which='first')
print(f"Switch OUT at: {switch_out:.2f}s (end of last Chorus)")
print(f"Switch IN at: {switch_in:.2f}s (first Verse)")


def download_song(artist, title, out_dir='audio'):
    os.makedirs(out_dir, exist_ok=True)
    search_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "default_search": "ytsearch5",
    }
    download_opts = {
        "format": "bestaudio/bestaudio*/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "outtmpl": f"{out_dir}/%(title)s.%(ext)s",
        "quiet": True,
        "no_warnings": True,
        "js_runtimes": {"node": {}},
        "remote_components": {"ejs": {"source": "github"}},
    }
    query = f"ytsearch5:{artist} {title}"
    with yt_dlp.YoutubeDL(search_opts) as ydl:
        info = ydl.extract_info(query, download=False)
        if not info.get("entries"):
            raise ValueError(f"No results for {artist} - {title}")
        best = max(info["entries"], key=lambda e: __import__('difflib').SequenceMatcher(
            None, f"{artist} {title}".lower(), e.get("title","").lower()).ratio())
        url = f"https://www.youtube.com/watch?v={best['id']}"
    with yt_dlp.YoutubeDL(download_opts) as ydl:
        ydl.extract_info(url, download=True)
    files = [f for f in os.listdir(out_dir) if f.endswith('.wav')]
    return os.path.join(out_dir, max(files, key=lambda f: os.path.getmtime(os.path.join(out_dir, f))))

def mix_songs(manifest, song_id_a, song_id_b, crossfade_ms=4000, out_path='mix_output.wav'):
    row_a = manifest[manifest['SONG_ID'] == song_id_a].iloc[0]
    row_b = manifest[manifest['SONG_ID'] == song_id_b].iloc[0]

    # Get switch points
    segs_a = get_segments(row_a)
    segs_b = get_segments(row_b)

    switch_out = find_switch_point(segs_a, 'Chorus', 'last') or find_switch_point(segs_a, 'Verse', 'last')
    switch_in  = find_switch_point(segs_b, 'Verse', 'first') or find_switch_point(segs_b, 'Chorus', 'first')

    if switch_out is None or switch_in is None:
        raise ValueError("Could not find switch points in one of the songs")

    print(f"Song A: {row_a['ARTIST']} - {row_a['SONG_TITLE']}")
    print(f"  Switching out at {switch_out:.2f}s (last Chorus)")
    print(f"Song B: {row_b['ARTIST']} - {row_b['SONG_TITLE']}")
    print(f"  Switching in at {switch_in:.2f}s (first Verse)")

    # Download both songs
    print("Downloading Song A...")
    path_a = download_song(row_a['ARTIST'], row_a['SONG_TITLE'])
    print("Downloading Song B...")
    path_b = download_song(row_b['ARTIST'], row_b['SONG_TITLE'])

    # Load audio
    audio_a = AudioSegment.from_wav(path_a)
    audio_b = AudioSegment.from_wav(path_b)

    # Cut: Song A up to switch point, Song B from switch_in onward
    part_a = audio_a[:int(switch_out * 1000)]
    part_b = audio_b[int(switch_in * 1000):]

    # Crossfade
    mix = part_a.append(part_b, crossfade=crossfade_ms)
    mix.export(out_path, format='wav')
    print(f"Mix saved to {out_path}")
    return out_path

# Test it — pick two songs from the manifest
print(manifest[['SONG_ID', 'ARTIST', 'SONG_TITLE']].head(10))

out = mix_songs(manifest, song_id_a=2, song_id_b=10)
