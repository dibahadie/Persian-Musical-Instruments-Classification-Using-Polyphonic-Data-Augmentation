# split_audio_random_m.py
from __future__ import annotations

import math
import random
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".aiff", ".alac"}

def _run_ffmpeg(args: List[str]) -> None:
    subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def _run_ffprobe(args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def ffprobe_duration_seconds(path: Path) -> float:
    """
    Returns duration in seconds using ffprobe.
    """
    cp = _run_ffprobe([
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ])
    return float(cp.stdout.strip())

def iter_audio_files(root: Path, recursive: bool = False) -> Iterable[Path]:
    if recursive:
        yield from (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    else:
        yield from (p for p in root.glob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)

def extract_chunk_wav(
    src_file: Path,
    dst_wav: Path,
    start_sec: float,
    dur_sec: float,
) -> None:
    """
    Extract a single chunk [start_sec, start_sec+dur_sec) from src_file to dst_wav,
    always encoding to WAV (PCM 16-bit).
    """
    args = [
        "ffmpeg", "-hide_banner", "-nostdin", "-y",
        "-ss", str(start_sec),     # -ss before -i for speed
        "-i", str(src_file),
        "-t", str(dur_sec),
        "-acodec", "pcm_s16le",    # force 16-bit PCM WAV
        # "-ar", "44100",          # uncomment to force sample rate
        # "-ac", "1",              # uncomment to force mono
        str(dst_wav),
    ]
    _run_ffmpeg(args)

def split_random_m_chunks_from_file(
    src_file: Path,
    dst_dir: Path,
    segment_time: float = 5.0,
    m: int = 3,
    rename_base: str = "sample",
    rng: Optional[random.Random] = None,
    start_index: int = 0,
) -> Tuple[List[Path], int]:
    """
    For a single audio file, randomly pick m segments of length segment_time (in seconds),
    extract only those, and place them DIRECTLY under dst_dir, naming them sequentially:

        {rename_base}{start_index}.wav, {rename_base}{start_index+1}.wav, ...

    Returns (list_of_created_paths, next_start_index).
    """
    rng = rng or random.Random()
    dst_dir.mkdir(parents=True, exist_ok=True)

    total = ffprobe_duration_seconds(src_file)
    if total <= 0:
        print(f"⚠️  Skipping (zero duration): {src_file}")
        return [], start_index

    num_segments = math.ceil(total / segment_time)
    if num_segments <= 0:
        print(f"⚠️  Skipping (no segments): {src_file}")
        return [], start_index

    k = min(m, num_segments)
    indices = list(range(num_segments))
    rng.shuffle(indices)
    pick = sorted(indices[:k])

    created: List[Path] = []
    idx = start_index
    for seg_idx in pick:
        start = seg_idx * segment_time
        dur = min(segment_time, max(0.0, total - start))  # last chunk may be shorter
        if dur <= 0:
            continue
        out_path = dst_dir / f"{rename_base}{idx}.wav"
        extract_chunk_wav(src_file, out_path, start_sec=start, dur_sec=dur)
        created.append(out_path)
        idx += 1

    print(f"✅ {src_file.name}: picked {len(created)} of {num_segments} segments -> {dst_dir}")
    return created, idx

def split_all_audio_random_m(
    source_dir: str | Path,
    destination_dir: str | Path,
    segment_time: float = 5.0,
    m: int = 3,
    recursive: bool = False,
    rename_base: str = "sample",
    seed: Optional[int] = None,
) -> None:
    """
    For every audio in source_dir, randomly select m segments of length segment_time and export them
    into a SINGLE shared destination folder as WAV files, named sequentially across ALL files:

        sample0.wav, sample1.wav, sample2.wav, ...

    Args:
        source_dir: where to read audio files from.
        destination_dir: single folder where ALL chunks are written.
        segment_time: seconds per chunk (default 5.0).
        m: number of chunks to keep per file (randomly selected).
        recursive: whether to search subfolders.
        rename_base: base name of chunk files (default "sample").
        seed: optional integer for reproducible random selection.
    """
    src_root = Path(source_dir).expanduser().resolve()
    dst_root = Path(destination_dir).expanduser().resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed) if seed is not None else random.Random()

    global_idx = 0
    for audio_path in iter_audio_files(src_root, recursive=recursive):
        try:
            _, global_idx = split_random_m_chunks_from_file(
                audio_path,
                dst_root,                   # single shared folder
                segment_time=segment_time,
                m=m,
                rename_base=rename_base,
                rng=rng,
                start_index=global_idx,     # keep naming sequential across files
            )
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed on {audio_path}\n{e.stderr}")

if __name__ == "__main__":
    # Example:
    split_all_audio_random_m(
        "input_audio",
        "output_chunks_wav",
        segment_time=5.0,
        m=5,
        recursive=False,
        rename_base="sample",
        seed=42
    )
    
