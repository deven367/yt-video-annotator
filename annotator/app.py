from pathlib import Path

import streamlit as st

from annotator.playback import playback_state, seek_to
from annotator.utils import (
    annotate2,
    df_from_result,
    filter_blank_audio,
    find_word_timestamp,
    generate_srt,
    get_audio,
    get_v_from_url,
    write_srt,
)

st.set_page_config(layout="wide")

SRT_PATH = Path("srt")
SRT_PATH.mkdir(exist_ok=True)


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


@st.cache_data(ttl=3600, max_entries=32)
def caption_from_url(url: str):
    """
    Download audio from URL and transcribe it.

    Uses caching to avoid re-downloading and re-transcribing the same video.
    """
    try:
        video_id = get_v_from_url(url)
        with get_audio(url) as (audio_path, was_cached):
            result = filter_blank_audio(annotate2(str(audio_path)))
    except Exception as error:
        st.error(f"Could not process video: {error}")
        st.stop()

    if was_cached:
        st.info(f"Using cached audio for video: {video_id}")
    else:
        st.success(f"Audio downloaded and converted: {audio_path.name}")


    if not result:
        st.warning("No speech detected in the audio.")
        return audio_path.name, None

    df = df_from_result(result)
    return audio_path.name, df


def main():
    url, name = None, None
    make_sidebar()
    place = "https://www.youtube.com/watch?v=C0DPdy98e4c&ab_channel=SimonYapp"
    col1, col2 = st.columns([1.2, 1])

    with col1:
        url = st.text_input("Enter URL for the YT video", place)
        playback = playback_state("search_playback", url)
        with st.container(key=f"search_player_{playback['revision']}"):
            st.video(
                url, start_time=playback["start_time"], autoplay=playback["autoplay"]
            )

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
            source = (url, tuple(words))
            if st.session_state.get("word_results", {}).get("source") != source:
                st.session_state.pop("word_results", None)

            if st.button("Get Timestamps") and words:
                st.session_state.pop("word_results", None)
                with st.spinner("Processing video..."):
                    audio_src, df = caption_from_url(url)
                    if df is not None:
                        st.session_state["word_results"] = {
                            "source": source,
                            "times": find_word_timestamp(df, *words),
                        }

            result = st.session_state.get("word_results")
            if result is not None:
                st.markdown("### Results")
                for word_index, (word, timestamps) in enumerate(zip(words, result["times"])):
                    if timestamps:
                        st.success(f"**{word}**")
                        with st.container(horizontal=True, gap="small"):
                            for match_index, timestamp in enumerate(timestamps):
                                st.button(
                                    timestamp,
                                    key=f"word_match_{word_index}_{match_index}",
                                    on_click=seek_to,
                                    args=("search_playback", url, timestamp),
                                    help=f"Play {word} from {timestamp}",
                                    type="tertiary",
                                )
                    else:
                        st.warning(f"**{word}** not found in the video")

        else:
            if st.button("Generate SRT"):
                with st.spinner("Processing video..."):
                    audio_src, df = caption_from_url(url)

                    if df is not None:
                        name = Path(audio_src).stem
                        s = generate_srt(df)
                        write_srt(s, SRT_PATH / name)
                        st.success("SRT file generated!")

        if name is not None:
            st.download_button(
                "Download SRT",
                s,
                file_name=f"{name}.srt",
                mime="application/x-subrip",
            )


if __name__ == "__main__":
    main()
