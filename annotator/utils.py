import whisper
import datetime
import pandas as pd
import numpy as np
import subprocess
from fastcore.foundation import working_directory
from pathlib import Path


def start_app():
    subprocess.run(["streamlit", "run", "app.py"])


def get_audio(url: str):
    audio_path = Path("./audio")
    if not audio_path.exists():
        audio_path.mkdir(exist_ok=True)
    with working_directory(audio_path):
        # subprocess.run(['youtube-dl', '-F', 'bestaudio[ext=m4a]', url])
        subprocess.run(["youtube-dl", "-x", "--audio-format", "mp3", url])


def annotate(audio_src, model_size="tiny"):
    model = whisper.load_model(model_size, device="cpu")
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
    for word in words:
        vals = df["text"].str.find(word).values
        arr = np.where(vals > 1)
        times = df.iloc[arr]["start"].values
        for t in times:
            t = t.split(".")[:-1]
            print(f"{word} is said on {t} timestamp")


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
