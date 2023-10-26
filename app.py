from pathlib import Path

import numpy as np
import streamlit as st
from fastcore.xtras import globtastic

from annotator.utils import (
    Annotator,
    df_from_result,
    find_word_timestamp,
    generate_srt,
    get_audio,
    get_v_from_url,
    working_directory,
    write_srt,
)

st.set_page_config(layout="wide")


SRT_PATH = Path("srt")
if not SRT_PATH.exists():
    SRT_PATH.mkdir()

AUDIO_PATH = Path("./audio")
if not AUDIO_PATH.exists():
    AUDIO_PATH.mkdir()


def make_sidebar():
    with st.sidebar:
        st.markdown("## yt-video-annotator")
        st.write("Link to the GitHub repo")


# @st.cache_resource
def caption_from_url(url):
    audio_src = get_audio(url)
    v = get_v_from_url(url)
    audio_src = globtastic(AUDIO_PATH, file_glob="*.wav", file_re=v)[0]
    ann = Annotator(audio_src)
    result = ann.annotate_quantize(audio_src)
    df = df_from_result(result)
    return audio_src, df


def main():
    url, name = None, None
    make_sidebar()
    place = "https://www.youtube.com/watch?v=C0DPdy98e4c&ab_channel=SimonYapp"
    col1, col2 = st.columns([1.2, 1])
    with col1:
        url = st.text_input("Enter URL for the YT video", place)
        st.video(url)

    with col2:
        default_opt = "Search for words"
        opt = st.radio(
            "What do you wish to do?",
            [default_opt, "Generate subtitles for the entire video"],
        )
        if opt == default_opt:
            st.markdown("### Search for words in the video")
            words = st.text_input("Enter words separated by a comma")
            words = words.split(",")

            if st.button("Get Timestamps"):
                audio_src, df = caption_from_url(url)
                times = find_word_timestamp(df, *words)
                times = np.asarray(times).reshape(len(words), -1)
                # st.write(times)
                for i, word in enumerate(words):
                    st.write(f"{word} is said on {times[i].flatten()} timestamp")

        else:
            if st.button("Generate SRT"):
                audio_src, df = caption_from_url(url)
                name = Path(audio_src).stem
                s = generate_srt(df)
                with working_directory(SRT_PATH):
                    write_srt(s, name)

        if name is not None:
            with working_directory(SRT_PATH):
                key = get_v_from_url(url)
                srt = globtastic(".", file_glob="*.srt", file_re=key)[0]
                with open(srt) as f:
                    st.download_button("Download SRT", f, file_name=f"{name}.srt")

    # subprocess.run(['rm', '-rf', 'audio'])
    # subprocess.run(['rm', '-rf', 'srt'])


if __name__ == "__main__":
    main()
