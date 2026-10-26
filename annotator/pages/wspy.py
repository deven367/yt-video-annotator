"""
Audio Transcription Page

Transcribe uploaded audio files and YouTube videos using Whisper.
"""

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st

from annotator.playback import playback_state, seek_to
from annotator.utils import (
    annotate2,
    convert_to_whisper_format,
    filter_blank_audio,
    get_audio,
    get_v_from_url,
)

st.set_page_config(layout="wide", page_title="Audio Transcription")


def format_timestamp(centiseconds: int) -> str:
    """Convert centiseconds to HH:MM:SS format."""
    seconds = centiseconds // 100
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def render_result(key, source):
    result = st.session_state.get(key)
    if result is None:
        return
    if result["source"] != source:
        del st.session_state[key]
        return
    segments = result["segments"]
    if not segments:
        st.warning("No speech detected in the audio.")
        return

    st.markdown("### Transcription Results")
    full_text = " ".join(seg.text.strip() for seg in segments)
    text = st.text_area(
        "Full Transcription", full_text, height=200, key=f"{key}_text_{source}"
    )
    with st.expander(
        "View Timestamped Segments", key=f"{key}_segments", on_change="rerun"
    ):
        for index, seg in enumerate(segments):
            start = format_timestamp(seg.t0)
            end = format_timestamp(seg.t1)
            timestamp, text_column = st.columns([1, 3])
            with timestamp:
                st.button(
                    start,
                    key=f"{key}_segment_{index}",
                    on_click=seek_to,
                    args=(f"{key}_playback", source, seg.t0 // 100),
                    help=f"Play from {start}",
                    type="tertiary",
                )
            text_column.markdown(f"**- {end}** {seg.text.strip()}")
    st.download_button(
        "Download Transcription", text, file_name=result["filename"],
        mime="text/plain", key=f"{key}_download",
    )


def main():
    st.title("Audio Transcription")
    st.markdown("Transcribe YouTube videos or upload audio files for transcription.")

    # Sidebar for model selection
    with st.sidebar:
        st.markdown("## Settings")
        model_size = st.selectbox(
            "Model Size",
            options=["tiny", "base", "small", "medium", "large-v3"],
            index=1,  # Default to 'base'
            help="Larger models are more accurate but slower"
        )
        st.info(
            "**Model Info:**\n"
            "- **tiny**: Fastest, least accurate\n"
            "- **base**: Good balance (recommended)\n"
            "- **small**: Better accuracy\n"
            "- **medium**: High accuracy\n"
            "- **large-v3**: Best accuracy, slowest"
        )

    # Tab selection for input method
    tab1, tab2 = st.tabs(["YouTube URL", "Upload Audio"])

    with tab1:
        st.markdown("### Transcribe from YouTube")
        url = st.text_input(
            "Enter YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            key="yt_url"
        )

        source = (url, model_size)
        if st.session_state.get("yt_result", {}).get("source") != source:
            st.session_state.pop("yt_result", None)
        playback = playback_state("yt_result_playback", source)
        if url:
            col1, col2 = st.columns([1, 1])

            with col1:
                with st.container(key=f"yt_player_{playback['revision']}"):
                    st.video(
                        url, start_time=playback["start_time"],
                        autoplay=playback["autoplay"],
                    )

            with col2:
                if st.button("Start Transcription", key="transcribe_yt"):
                    st.session_state.pop("yt_result", None)
                    with st.spinner("Processing audio from YouTube..."):
                        try:
                            video_id = get_v_from_url(url)

                            with get_audio(url) as (audio_path, was_cached):
                                if was_cached:
                                    st.info(f"Using cached audio for: {video_id}")
                                else:
                                    st.success(f"Audio downloaded and converted: {audio_path.name}")

                                with st.spinner("Transcribing... This may take a while."):
                                    segments = annotate2(str(audio_path), model_size=model_size)
                                    segments = filter_blank_audio(segments)

                            st.session_state["yt_result"] = {
                                "source": source,
                                "segments": segments,
                                "filename": f"{video_id}_transcription.txt",
                            }
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                render_result("yt_result", source)

    with tab2:
        st.markdown("### Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            key="audio_upload"
        )

        source = (
            (uploaded_file.name, sha256(uploaded_file.getbuffer()).hexdigest(), model_size)
            if uploaded_file is not None else None
        )
        if st.session_state.get("upload_result", {}).get("source") != source:
            st.session_state.pop("upload_result", None)
        playback = playback_state("upload_result_playback", source)
        if uploaded_file is not None:
            with st.container(key=f"upload_player_{playback['revision']}"):
                st.audio(
                    uploaded_file, start_time=playback["start_time"],
                    autoplay=playback["autoplay"],
                )

            if st.button("Start Transcription", key="transcribe_upload"):
                st.session_state.pop("upload_result", None)
                with st.spinner("Processing audio..."), TemporaryDirectory() as temp_dir:
                    try:
                        temp_path = Path(temp_dir) / f"upload{Path(uploaded_file.name).suffix}"
                        converted_path = Path(temp_dir) / "converted.wav"
                        temp_path.write_bytes(uploaded_file.getbuffer())
                        if not convert_to_whisper_format(temp_path, converted_path):
                            raise RuntimeError("Could not convert the uploaded audio.")
                        st.info("Audio converted to 16kHz WAV format.")

                        with st.spinner("Transcribing... This may take a while."):
                            segments = annotate2(str(converted_path), model_size=model_size)
                            segments = filter_blank_audio(segments)

                        st.session_state["upload_result"] = {
                            "source": source,
                            "segments": segments,
                            "filename": f"{uploaded_file.name}_transcription.txt",
                        }
                    except Exception as e:
                        st.error(f"Error during transcription: {str(e)}")
            render_result("upload_result", source)


if __name__ == "__main__":
    main()
