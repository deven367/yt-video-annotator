import whisper
import datetime
import pandas as pd
import numpy as np


def annotate(audio_src, model_size="tiny"):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_src)
    return result


def get_time(seconds):
    return "{:0>8}".format(str(datetime.timedelta(seconds=seconds)))


def df_from_result(result):
    df = pd.json_normalize(result)
    df["start"] = df["start"].apply(get_time)
    df["end"] = df["end"].apply(get_time)
    return df


def find_word_timestamp(df, *words):
    for word in words:
        vals = df["text"].str.find(word).values
        arr = np.where(vals > 1)
        times = df.iloc[arr]['start'].values
        for t in times:
            t = t.split('.')[:-1]
            print(f'{word} is said on {t} timestamp')
