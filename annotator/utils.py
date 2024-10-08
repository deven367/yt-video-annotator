import datetime
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import whisper
from fastcore.foundation import L, working_directory


def start_app():
    subprocess.run(["streamlit", "run", "app.py"])


def get_audio(url: str):
    audio_path = Path("./audio")
    with working_directory(audio_path):
        # subprocess.run(['youtube-dl', '-F', 'bestaudio[ext=m4a]', url])
        subprocess.run(["yt-dlp", "-x", "--audio-format", "wav", url])


def get_v_from_url(url):
    _, val = url.split("?v=")
    return val.split("&")[0]


def annotate(audio_src, model_size="tiny"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_size, device=device)
    result = model.transcribe(audio_src)
    return result


def get_time(seconds):
    return "{:0>8}".format(str(datetime.timedelta(seconds=seconds)))


def df_from_result(result):
    df = pd.json_normalize(result["segments"])
    df["start"] = df["start"].apply(get_time)
    df["end"] = df["end"].apply(get_time)
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
