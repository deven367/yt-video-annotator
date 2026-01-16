import datetime
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# import torch
# import whisper
from fastcore.foundation import L, working_directory
from pywhispercpp.model import Model

ROOT_PATH = Path(__file__).parent.parent
AUDIO_PATH = Path("./audio")


def start_app():
    subprocess.run(["streamlit", "run", "app.py"])


def ensure_audio_dir():
    """Ensure audio directory exists."""
    if not AUDIO_PATH.exists():
        AUDIO_PATH.mkdir(parents=True)


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
    if converted_path.exists():
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
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(input_path),
                "-ar", "16000",  # 16kHz sample rate
                "-ac", "1",      # mono
                "-c:a", "pcm_s16le",  # 16-bit PCM
                str(output_path)
            ],
            check=True,
            capture_output=True
        )
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
    ensure_audio_dir()
    video_id = get_v_from_url(url)
    
    # Check for cached converted audio
    converted_path = AUDIO_PATH / f"{video_id}_converted.wav"
    if not force_download and converted_path.exists():
        return converted_path, True
    
    # Download audio from YouTube
    with working_directory(AUDIO_PATH):
        subprocess.run(["yt-dlp", "-x", "--audio-format", "wav", url])
    
    # Find the downloaded file
    from fastcore.xtras import globtastic
    audio_files = globtastic(AUDIO_PATH, file_re=video_id)
    
    if not audio_files:
        raise FileNotFoundError(f"Could not find downloaded audio for video {video_id}")
    
    downloaded_path = Path(audio_files[0])
    
    # Convert to Whisper-compatible format
    if convert_to_whisper_format(downloaded_path, converted_path):
        # Remove original downloaded file to save space
        if downloaded_path != converted_path and downloaded_path.exists():
            downloaded_path.unlink()
        return converted_path, False
    else:
        # If conversion fails, return the original file
        return downloaded_path, False


def get_v_from_url(url:str):
    try:
        _, val = url.split("?v=")
        return val.split("&")[0]
    except ValueError:
        if url.startswith("https://youtu.be/"):
            to_parse = url.split(".be/")[1]
            return to_parse.split("?")[0]


# def annotate(audio_src, model_size="tiny"):
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     model = whisper.load_model(model_size, device=device)
#     result = model.transcribe(audio_src)
#     return result

def annotate2(audio_src, model_size="base", single_segment=False):
    """
    Transcribe audio using pywhispercpp.
    
    Args:
        audio_src: Path to the audio file
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        single_segment: If True, process entire audio as one segment. 
                       Set to False (default) for better results on longer audio.
    
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


def get_time(seconds):
    return "{:0>8}".format(str(datetime.timedelta(seconds=seconds)))


def df_from_result(result):
    # df = pd.json_normalize(result["segments"])
    # df["start"] = df["start"].apply(get_time)
    # df["end"] = df["end"].apply(get_time)
    # return df

    d = defaultdict(list)
    for segment in result:
        d['start'].append(segment.t0)
        d['end'].append(segment.t1)
        d['text'].append(segment.text)
    df = pd.DataFrame(d)
    df['start'] = df['start'].apply(get_time)
    df['end'] = df['end'].apply(get_time)

    return df

def find_word_timestamp(df, *words):
    timestamps = L()
    for word in words:
        vals = df["text"].str.find(word).values
        arr = np.where(vals > 1)
        times = list(df.iloc[arr]["start"].values)
        nt = L(times).map(lambda x: x.split(".")[:-1])
        timestamps.append(nt)
    return timestamps


def generate_srt(df):
    s = ""
    for i, (start, end, text) in enumerate(df[["start", "end", "text"]].values):
        start = start.replace(".", ",")
        end = end.replace(".", ",")
        s += f"{i}\n"
        s += f"{start} --> {end}\n"
        s += f"{text.strip()}\n\n"
    return s


def write_srt(s, name):
    with open(f"{name}.srt", "w") as f:
        f.write(s)
        f.close()
