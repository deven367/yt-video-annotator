import subprocess
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
import streamlit as st
from pywhispercpp.constants import AVAILABLE_MODELS
from pywhispercpp.model import Segment
from streamlit.testing.v1 import AppTest

from annotator import utils

ROOT = Path(__file__).resolve().parents[1]
SEGMENTS = [Segment(0, 125, "hello world"), Segment(125, 250, " hello again")]


def test_model_choices_are_supported():
    page = AppTest.from_file(str(ROOT / "pages" / "wspy.py")).run()
    assert not page.exception
    assert set(page.selectbox[0].options) <= set(AVAILABLE_MODELS)


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    st.cache_data.clear()
    with patch.object(utils, "get_audio", return_value=(tmp_path / "audio.wav", True)):
        with patch.object(utils, "annotate2", return_value=SEGMENTS):
            yield AppTest.from_file(str(ROOT / "app.py")).run()
    st.cache_data.clear()


def test_search_handles_uneven_and_missing_matches(app):
    app.text_input[1].set_value("hello, world, missing")
    app.button[0].click().run()
    assert not app.exception
    assert "hello** found at: 00:00:00, 00:00:01" in app.success[0].value
    assert "world** found at: 00:00:00" in app.success[1].value
    assert "missing** not found" in app.warning[0].value


def test_srt_is_written_and_offered_for_download(app, tmp_path):
    app.radio[0].set_value("Generate subtitles for the entire video").run()
    app.button[0].click().run()
    assert not app.exception
    assert (
        (tmp_path / "srt" / "audio.srt")
        .read_text()
        .startswith("1\n00:00:00,000 --> 00:00:01,250\nhello world\n\n")
    )
    assert len(app.get("download_button")) == 1


@pytest.mark.parametrize(
    "error", [ValueError("Invalid URL"), subprocess.CalledProcessError(1, "yt-dlp")]
)
def test_video_errors_are_displayed_and_can_be_retried(app, error):
    with patch.object(utils, "get_audio", side_effect=error):
        app.text_input[1].set_value("hello")
        app.button[0].click().run()
    assert not app.exception
    assert "Could not process video" in app.error[0].value
    app.button[0].click().run()
    assert not app.exception
    assert "hello** found at:" in app.success[0].value


@pytest.mark.parametrize(
    "filename", ["sample.wav", "C0DPdy98e4c_converted.wav", "../outside.wav"]
)
def test_upload_does_not_overwrite_or_delete_existing_files(
    tmp_path, monkeypatch, filename
):
    monkeypatch.chdir(tmp_path)
    audio_dir = tmp_path / "audio"
    monkeypatch.setattr(utils, "AUDIO_PATH", audio_dir)
    audio_dir.mkdir()
    cached = audio_dir / "C0DPdy98e4c_converted.wav"
    cached.write_bytes(b"cached audio")
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"unrelated file")
    upload = BytesIO(b"uploaded audio")
    upload.name = filename
    paths = []

    def convert(source, output):
        assert source.read_bytes() == b"uploaded audio"
        paths.extend([source, output])
        output.write_bytes(b"converted audio")
        return True

    def transcribe(path, **kwargs):
        assert Path(path).read_bytes() == b"converted audio"
        return SEGMENTS

    with patch.object(st, "file_uploader", return_value=upload):
        with patch.object(utils, "convert_to_whisper_format", side_effect=convert):
            with patch.object(utils, "annotate2", side_effect=transcribe):
                page = AppTest.from_file(str(ROOT / "pages" / "wspy.py")).run()
                page.button(key="transcribe_upload").click().run()
    assert not page.exception
    assert not page.error
    assert page.text_area[0].value == "hello world hello again"
    assert len(paths) == 2
    assert all(not path.exists() for path in paths)
    assert cached.read_bytes() == b"cached audio"
    assert outside.read_bytes() == b"unrelated file"


def test_upload_works_without_audio_directory_and_cleans_up_on_failure(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(utils, "AUDIO_PATH", tmp_path / "missing-audio")
    upload = BytesIO(b"uploaded audio")
    upload.name = "sample.wav"
    paths = []

    def fail(source, output):
        paths.append(source)
        return False

    with patch.object(st, "file_uploader", return_value=upload):
        with patch.object(utils, "convert_to_whisper_format", side_effect=fail):
            with patch.object(utils, "annotate2") as transcribe:
                page = AppTest.from_file(str(ROOT / "pages" / "wspy.py")).run()
                page.button(key="transcribe_upload").click().run()
    assert not page.exception
    assert "Could not convert the uploaded audio" in page.error[0].value
    transcribe.assert_not_called()
    assert paths and not paths[0].exists()


@pytest.mark.parametrize("method", ["youtube", "upload"])
def test_transcription_survives_reruns_and_tracks_source_and_model(
    tmp_path, monkeypatch, method
):
    monkeypatch.chdir(tmp_path)
    upload = BytesIO(b"first recording")
    upload.name = "recording.wav"
    with (
        patch.object(st, "file_uploader", return_value=upload if method == "upload" else None),
        patch.object(utils, "get_audio", return_value=(tmp_path / "audio.wav", True)),
        patch.object(utils, "convert_to_whisper_format", return_value=True),
        patch.object(utils, "annotate2", return_value=SEGMENTS) as transcribe,
    ):
        page = AppTest.from_file(str(ROOT / "pages" / "wspy.py")).run()
        if method == "youtube":
            page.text_input[0].set_value("https://youtu.be/C0DPdy98e4c").run()
        button = "transcribe_yt" if method == "youtube" else "transcribe_upload"
        page.button(key=button).click().run()
        assert page.text_area[0].value == "hello world hello again"
        page.text_area[0].set_value("edited transcript").run()
        page.run()
        assert not page.exception
        assert page.text_area[0].value == "edited transcript"
        assert len(page.get("download_button")) == 1
        assert transcribe.call_count == 1

        page.selectbox[0].set_value("tiny").run()
        assert not page.text_area
        assert not page.get("download_button")
        page.button(key=button).click().run()
        assert page.text_area[0].value == "hello world hello again"
        if method == "youtube":
            page.text_input[0].set_value("https://youtu.be/rdzKbICr1A0").run()
        else:
            upload.seek(0)
            upload.write(b"other recording")
            page.run()
        assert not page.text_area
        assert not page.get("download_button")
        assert transcribe.call_count == 2
