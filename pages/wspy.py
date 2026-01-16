"""
Live Stream Transcription Page

This page provides real-time transcription capabilities for audio streams
and YouTube videos using Whisper.
"""

import subprocess
from pathlib import Path

import streamlit as st
from fastcore.xtras import globtastic

from annotator.utils import annotate2, get_audio, get_v_from_url

st.set_page_config(layout="wide", page_title="Live Transcription")

AUDIO_PATH = Path("./audio")
if not AUDIO_PATH.exists():
    AUDIO_PATH.mkdir()


def convert_to_wav(input_path: Path, output_path: Path) -> bool:
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
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def format_timestamp(centiseconds: int) -> str:
    """Convert centiseconds to HH:MM:SS format."""
    seconds = centiseconds // 100
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def filter_blank_audio(segments: list) -> list:
    """Filter out BLANK_AUDIO segments from results."""
    return [seg for seg in segments if "[BLANK_AUDIO]" not in seg.text]


def main():
    st.title("🎙️ Audio Transcription")
    st.markdown("Transcribe YouTube videos or upload audio files for transcription.")

    # Sidebar for model selection
    with st.sidebar:
        st.markdown("## Settings")
        model_size = st.selectbox(
            "Model Size",
            options=["tiny", "base", "small", "medium", "large"],
            index=1,  # Default to 'base'
            help="Larger models are more accurate but slower"
        )
        st.info(
            "💡 **Model Info:**\n"
            "- **tiny**: Fastest, least accurate\n"
            "- **base**: Good balance (recommended)\n"
            "- **small**: Better accuracy\n"
            "- **medium**: High accuracy\n"
            "- **large**: Best accuracy, slowest"
        )

    # Tab selection for input method
    tab1, tab2 = st.tabs(["📺 YouTube URL", "📁 Upload Audio"])

    with tab1:
        st.markdown("### Transcribe from YouTube")
        url = st.text_input(
            "Enter YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            key="yt_url"
        )

        if url:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.video(url)

            with col2:
                if st.button("🎯 Start Transcription", key="transcribe_yt"):
                    with st.spinner("Downloading audio from YouTube..."):
                        try:
                            get_audio(url)
                            video_id = get_v_from_url(url)
                            
                            # Find the downloaded audio file
                            audio_files = globtastic(AUDIO_PATH, file_re=video_id)
                            
                            if audio_files:
                                audio_file = audio_files[0]
                                st.success(f"Audio downloaded: {Path(audio_file).name}")
                                
                                with st.spinner("Transcribing... This may take a while."):
                                    # Use the working annotate2 function from utils
                                    segments = annotate2(str(audio_file), model_size=model_size)
                                    segments = filter_blank_audio(segments)
                                
                                if segments:
                                    st.success("✅ Transcription complete!")
                                    
                                    # Display results
                                    st.markdown("### Transcription Results")
                                    
                                    # Full text
                                    full_text = " ".join(seg.text.strip() for seg in segments)
                                    st.text_area("Full Transcription", full_text, height=200)
                                    
                                    # Timestamped segments
                                    with st.expander("📋 View Timestamped Segments"):
                                        for seg in segments:
                                            start = format_timestamp(seg.t0)
                                            end = format_timestamp(seg.t1)
                                            st.markdown(f"**[{start} - {end}]** {seg.text.strip()}")
                                    
                                    # Download button
                                    st.download_button(
                                        "📥 Download Transcription",
                                        full_text,
                                        file_name=f"{video_id}_transcription.txt",
                                        mime="text/plain"
                                    )
                                else:
                                    st.warning("No speech detected in the audio. The video may not contain speech or the audio quality may be too low.")
                            else:
                                st.error("Could not find downloaded audio file.")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

    with tab2:
        st.markdown("### Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            key="audio_upload"
        )

        if uploaded_file is not None:
            st.audio(uploaded_file)
            
            if st.button("🎯 Start Transcription", key="transcribe_upload"):
                # Save uploaded file temporarily
                temp_path = AUDIO_PATH / uploaded_file.name
                converted_path = AUDIO_PATH / f"converted_{Path(uploaded_file.name).stem}.wav"
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                with st.spinner("Processing audio..."):
                    try:
                        # Convert to proper format for Whisper (16kHz mono WAV)
                        audio_to_transcribe = temp_path
                        if convert_to_wav(temp_path, converted_path):
                            audio_to_transcribe = converted_path
                            st.info("Audio converted to 16kHz WAV format.")
                        
                        with st.spinner("Transcribing... This may take a while."):
                            # Use the working annotate2 function from utils
                            segments = annotate2(str(audio_to_transcribe), model_size=model_size)
                            segments = filter_blank_audio(segments)
                        
                        if segments:
                            st.success("✅ Transcription complete!")
                            
                            # Display results
                            st.markdown("### Transcription Results")
                            
                            # Full text
                            full_text = " ".join(seg.text.strip() for seg in segments)
                            st.text_area("Full Transcription", full_text, height=200)
                            
                            # Timestamped segments
                            with st.expander("📋 View Timestamped Segments"):
                                for seg in segments:
                                    start = format_timestamp(seg.t0)
                                    end = format_timestamp(seg.t1)
                                    st.markdown(f"**[{start} - {end}]** {seg.text.strip()}")
                            
                            # Download button
                            st.download_button(
                                "📥 Download Transcription",
                                full_text,
                                file_name=f"{uploaded_file.name}_transcription.txt",
                                mime="text/plain"
                            )
                        else:
                            st.warning("No speech detected in the audio. The file may not contain speech or the audio quality may be too low.")
                    except Exception as e:
                        st.error(f"Error during transcription: {str(e)}")
                    finally:
                        # Clean up temp files
                        if temp_path.exists():
                            temp_path.unlink()
                        if converted_path.exists():
                            converted_path.unlink()


if __name__ == "__main__":
    main()
