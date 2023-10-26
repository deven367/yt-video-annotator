import datetime
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import torch
    import whisper
except:
    print("torch and whisper not installed")
from fastcore.foundation import L, working_directory


def start_app():
    subprocess.run(["streamlit", "run", "app.py"])


def get_audio(url: str):
    audio_path = Path("./audio")
    with working_directory(audio_path):
        # subprocess.run(['youtube-dl', '-F', 'bestaudio[ext=m4a]', url])
        subprocess.run(["yt-dlp", "-x", "--audio-format", "wav", url])

def get_v_from_url(url):
    _, val = url.split('?v=')
    return val.split('&')[0]


class Annotator:

    def __init__(self, audio_src, model_size="tiny"):
        self.audio_src = audio_src
        self.model_size = model_size
        # self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # self.model = whisper.load_model(model_size, device=self.device)
        # self.result = self.model.transcribe(audio_src)
        # self.df = self.df_from_result()

    def annotate(audio_src, model_size="tiny"):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = whisper.load_model(model_size, device=device)
        result = model.transcribe(audio_src)
        return result

    def annotate_quantize(self, audio_src):
        cvrt = f"""ffmpeg -i XXX -ar 16000 -ac 1 -c:a pcm_s16le output.wav"""
        print(cvrt)
        cvrt = cvrt.split()
        l = list(map(lambda x: x.replace("XXX", audio_src), cvrt))
        print(l)
        stdout = subprocess.run(l, capture_output=True)
        print(stdout.stderr)

        cmd = "./main -m models/ggml-large-q5_0.bin -f output.wav"
        stdout = subprocess.run(cmd.split(), capture_output=True)
        print(stdout.stdout.decode('ascii'))
        print(stdout.stderr.decode('ascii'))


def get_time(seconds):
    return "{:0>8}".format(str(datetime.timedelta(seconds=seconds)))


def df_from_result(result):
    df = pd.json_normalize(result["segments"])
    df["start"] = df["start"].apply(get_time)
    df["end"] = df["end"].apply(get_time)
    return df


def find_word_timestamp(df, *words):
    l = L()
    for word in words:
        vals = df["text"].str.find(word).values
        arr = np.where(vals > 1)
        times = list(df.iloc[arr]["start"].values)
        nt = L(times).map(lambda x: x.split(".")[:-1])
        l.append(nt)
    return l



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
