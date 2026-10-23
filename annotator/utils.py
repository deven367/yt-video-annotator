import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Tuple
from urllib.parse import parse_qs, urlparse

import pandas as pd
from pywhispercpp.model import Model
from pywhispercpp.utils import to_timestamp

ROOT_PATH = Path(__file__).parent.parent
AUDIO_PATH = Path("./audio")


def start_app():
    subprocess.run(["streamlit", "run", "app.py"])


def ensure_audio_dir():
    """Ensure audio directory exists."""
    AUDIO_PATH.mkdir(parents=True, exist_ok=True)


def get_cached_audio(video_id: str) -> Optional[Path]:
    """
    Check if audio for a video already exists (converted WAV format).

    Args:
        video_id: YouTube video ID

    Returns:
        Path to the converted WAV file if it exists, None otherwise
    """
    ensure_audio_dir()
    converted_path = AUDIO_PATH / f"{video_id}_converted.wav"
    if converted_path.is_file() and converted_path.stat().st_size > 0:
        return converted_path
    return None


def convert_to_whisper_format(input_path: Path, output_path: Path) -> bool:
    """
    Convert audio file to 16kHz mono WAV format required by Whisper.

    Args:
        input_path: Path to input audio file
        output_path: Path to output WAV file

    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        with TemporaryDirectory(dir=output_path.parent) as temp_dir:
            temp_output = Path(temp_dir) / "converted.wav"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(input_path),
                    "-ar", "16000",  # 16kHz sample rate
                    "-ac", "1",      # mono
                    "-c:a", "pcm_s16le",  # 16-bit PCM
                    str(temp_output),
                ],
                check=True,
                capture_output=True,
            )
            temp_output.replace(output_path)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"FFmpeg conversion failed: {e}")
        return False


def get_audio(url: str, force_download: bool = False) -> Tuple[Path, bool]:
    """
    Download and convert audio from YouTube URL.

    Checks if audio already exists and is properly converted before downloading.

    Args:
        url: YouTube video URL
        force_download: If True, re-download even if cached

    Returns:
        Tuple of (path to converted audio file, whether it was cached)
    """
    video_id = get_v_from_url(url)
    ensure_audio_dir()

    converted_path = AUDIO_PATH / f"{video_id}_converted.wav"
    if not force_download:
        cached = get_cached_audio(video_id)
        if cached is not None:
            return cached, True

    with TemporaryDirectory(dir=AUDIO_PATH) as temp_dir:
        downloaded_path = Path(temp_dir) / "audio.wav"
        subprocess.run(
            [
                "yt-dlp", "--no-playlist", "-x", "--audio-format", "wav",
                "-o", str(Path(temp_dir) / "audio.%(ext)s"),
                "--", f"https://www.youtube.com/watch?v={video_id}",
            ],
            check=True,
        )
        if not convert_to_whisper_format(downloaded_path, converted_path):
            raise RuntimeError(f"Could not convert audio for video {video_id}")
    return converted_path, False


def get_v_from_url(url: str):
    parsed = urlparse(url.strip())
    video_id = ""
    if parsed.scheme in {"http", "https"}:
        if parsed.hostname in {"youtu.be", "www.youtu.be"}:
            video_id = parsed.path.strip("/")
        elif parsed.hostname in {
            "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
        }:
            if parsed.path == "/watch":
                video_id = parse_qs(parsed.query).get("v", [""])[0]
            elif parsed.path.startswith(("/shorts/", "/live/", "/embed/")):
                video_id = parsed.path.split("/")[2]
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Enter a valid YouTube video URL.")
    return video_id


def annotate2(audio_src, model_size="base", single_segment=False):
    """
    Transcribe audio using pywhispercpp.

    Args:
        audio_src: Path to the audio file
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large-v3')
        single_segment: Force a single segment per decoding window when True.

    Returns:
        List of transcription segments with timestamps
    """
    model = Model(model=model_size, single_segment=single_segment)
    result = model.transcribe(audio_src)
    return result


def filter_blank_audio(segments: list) -> list:
    """
    Filter out BLANK_AUDIO segments from transcription results.

    Args:
        segments: List of transcription segments

    Returns:
        Filtered list without BLANK_AUDIO segments
    """
    return [seg for seg in segments if "[BLANK_AUDIO]" not in seg.text]


def df_from_result(result):
    return pd.DataFrame(
        [
            (to_timestamp(seg.t0, separator="."), to_timestamp(seg.t1, separator="."), seg.text)
            for seg in result
        ],
        columns=["start", "end", "text"],
    )


def find_word_timestamp(df, *words):
    timestamps = []
    for word in words:
        times = df.loc[df["text"].str.contains(word, regex=False), "start"]
        timestamps.append([time.split(".")[0] for time in times])
    return timestamps


def generate_srt(df):
    s = ""
    for i, (start, end, text) in enumerate(df[["start", "end", "text"]].values, start=1):
        start = start.replace(".", ",")
        end = end.replace(".", ",")
        s += f"{i}\n"
        s += f"{start} --> {end}\n"
        s += f"{text.strip()}\n\n"
    return s


def write_srt(s, name):
    with open(f"{name}.srt", "w", encoding="utf-8") as f:
        f.write(s)
