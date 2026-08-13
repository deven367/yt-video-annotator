from pathlib import Path

import numpy as np
import streamlit as st

from annotator.utils import (
    annotate2,
    df_from_result,
    filter_blank_audio,
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


def make_sidebar():
    with st.sidebar:
        st.markdown("## yt-video-annotator")
        st.markdown(
            "[GitHub Repository](https://github.com/deven367/yt-video-annotator)"
        )
        st.markdown("---")
        st.markdown(
            "**About:** Annotate YouTube videos on-the-fly using Whisper speech recognition."
        )
        st.markdown(
            "**Features:**\n"
            "- Search for words in videos\n"
            "- Generate SRT subtitles\n"
            "- Download transcriptions"
        )


@st.cache_data
def caption_from_url(url: str):
    """
    Download audio from URL and transcribe it.
    
    Uses caching to avoid re-downloading and re-transcribing the same video.
    """
    video_id = get_v_from_url(url)
    
    # Download and convert audio (checks cache first)
    audio_path, was_cached = get_audio(url)
    
    if was_cached:
        st.info(f"Using cached audio for video: {video_id}")
    else:
        st.success(f"Audio downloaded and converted: {audio_path.name}")
    
    # Transcribe
    result = annotate2(str(audio_path))
    
    # Filter out blank audio segments
    result = filter_blank_audio(result)
    
    if not result:
        st.warning("No speech detected in the audio.")
        return str(audio_path), None
    
    df = df_from_result(result)
    return str(audio_path), df


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
            words = [w.strip() for w in words.split(",") if w.strip()]

            if st.button("Get Timestamps") and words:
                with st.spinner("Processing video..."):
                    audio_src, df = caption_from_url(url)
                    
                    if df is not None:
                        times = find_word_timestamp(df, *words)
                        times = np.asarray(times, dtype=object).reshape(len(words), -1)
                        
                        st.markdown("### Results")
                        for i, word in enumerate(words):
                            timestamps = times[i].flatten()
                            if len(timestamps) > 0:
                                st.success(f"**{word}** found at: {', '.join(map(str, timestamps))}")
                            else:
                                st.warning(f"**{word}** not found in the video")

        else:
            if st.button("Generate SRT"):
                with st.spinner("Processing video..."):
                    audio_src, df = caption_from_url(url)
                    
                    if df is not None:
                        name = Path(audio_src).stem
                        s = generate_srt(df)
                        with working_directory(SRT_PATH):
                            write_srt(s, name)
                        st.success("SRT file generated!")

        if name is not None:
            with working_directory(SRT_PATH):
                key = get_v_from_url(url)
                from fastcore.xtras import globtastic
                srt_files = globtastic(".", file_glob="*.srt", file_re=key)
                if srt_files:
                    srt = srt_files[0]
                    with open(srt) as f:
                        st.download_button(
                            "Download SRT",
                            f.read(),
                            file_name=f"{name}.srt",
                            mime="text/plain"
                        )


if __name__ == "__main__":
    main()
