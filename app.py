import streamlit as st
from annotator.utils import *
st.set_page_config(layout='wide')
from fastcore.xtras import globtastic
from pathlib import Path
import subprocess

SRT_PATH = Path('srt')
if not SRT_PATH.exists(): SRT_PATH.mkdir(exist_ok=True)

AUDIO_PATH = Path('./audio')
if not AUDIO_PATH.exists(): AUDIO_PATH.mkdir(exist_ok=True)


def make_sidebar():
    with st.sidebar:
        st.write('App')
        st.write('YouTube')


def main():
    make_sidebar()
    # st.write('This is it!')
    url = st.text_input('Enter URL for the YT video')

    if st.button('Generate SRT'):
        audio_src = get_audio(url)
        audio_src = globtastic(AUDIO_PATH, file_glob='*.mp3')[0]
        result = annotate(audio_src)
        df = df_from_result(result)

        # st.write(result.get('segments', 'wrong key'))
        st.write(df)
        name = Path(audio_src).stem
        s = generate_srt(df)
        with working_directory(SRT_PATH):
            write_srt(s, name)

        with working_directory(SRT_PATH):
            srt = globtastic('.', file_glob='*.srt')[0]
            with open(srt) as f:
                st.download_button('Download SRT', f, file_name=f'{name}.srt')


if __name__ == "__main__":
    main()