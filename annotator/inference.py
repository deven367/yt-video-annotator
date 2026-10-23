"""
Whisper inference module for audio transcription.

This module provides functions for transcribing audio files using
pywhispercpp (whisper.cpp Python bindings).
"""

from pathlib import Path
from typing import Optional, Union

from pywhispercpp.model import Model

from annotator.utils import ROOT_PATH


def load_model(model_size: str = "base") -> Model:
    """
    Load a Whisper model.

    Args:
        model_size: Size of the model to load. Options: 'tiny', 'base', 'small',
                   'medium', 'large-v3'. Default is 'base'.

    Returns:
        Model: Loaded pywhispercpp Model instance.
    """
    return Model(model=model_size)


def transcribe_audio(
    audio_path: Union[str, Path],
    model_size: str = "base",
    model: Optional[Model] = None,
) -> list:
    """
    Transcribe an audio file using Whisper.

    Args:
        audio_path: Path to the audio file to transcribe.
        model_size: Size of the model to use if model is not provided.
        model: Optional pre-loaded Model instance. If None, a new model will be loaded.

    Returns:
        list: List of transcription segments with timestamps.
    """
    if model is None:
        model = load_model(model_size)

    audio_path = str(audio_path)
    result = model.transcribe(audio_path)
    return result


def transcribe_to_text(
    audio_path: Union[str, Path],
    model_size: str = "base",
) -> str:
    """
    Transcribe an audio file and return plain text.

    Args:
        audio_path: Path to the audio file to transcribe.
        model_size: Size of the model to use.

    Returns:
        str: Full transcription text.
    """
    segments = transcribe_audio(audio_path, model_size)
    return " ".join(segment.text.strip() for segment in segments)


def get_local_model_path(filename: str = "ggml-base.bin") -> Path:
    """
    Get the path to a local GGML model file.

    Args:
        filename: Name of the model file.

    Returns:
        Path: Full path to the model file in the project root.
    """
    return ROOT_PATH / filename


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        print(f"Transcribing: {audio_file}")
        text = transcribe_to_text(audio_file)
        print(f"Transcription:\n{text}")
    else:
        print("Usage: python inference.py <audio_file>")
        print("\nAvailable model sizes: tiny, base, small, medium, large-v3")
