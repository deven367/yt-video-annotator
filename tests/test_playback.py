from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
import streamlit as st
from pywhispercpp.model import Segment
from streamlit.testing.v1 import AppTest

from annotator import utils

ROOT = Path(utils.__file__).resolve().parent
SEGMENTS = [
    Segment(0, 100, "hello zero"),
    Segment(360199, 360299, "hello hour"),
    Segment(360199, 360299, "hello duplicate"),
]


@pytest.mark.parametrize("method", ["search", "youtube", "upload"])
def test_seeking_preserves_results_without_reprocessing_and_tracks_source(
    tmp_path, monkeypatch, method
):
    monkeypatch.chdir(tmp_path)
    st.cache_data.clear()
    upload = BytesIO(b"recording")
    upload.name = "recording.wav"
    with (
        patch.object(st, "file_uploader", return_value=upload if method == "upload" else None),
        patch.object(utils, "get_audio", return_value=nullcontext((tmp_path / "audio.wav", True))) as download,
        patch.object(utils, "convert_to_whisper_format", return_value=True),
        patch.object(utils, "annotate2", return_value=SEGMENTS) as transcribe,
    ):
        script = ROOT / "app.py" if method == "search" else ROOT / "pages" / "wspy.py"
        page = AppTest.from_file(str(script)).run()
        if method == "search":
            page.text_input[1].set_value("hello,hello,missing")
            page.button[0].click().run()
            prefix = "word_match_0_"
            assert [button.label for button in page.button if button.key and button.key.startswith("word_match_")] == [
                "00:00:00", "01:00:01", "01:00:01",
                "00:00:00", "01:00:01", "01:00:01",
            ]
        else:
            if method == "youtube":
                page.text_input[0].set_value("https://youtu.be/C0DPdy98e4c").run()
            action = "transcribe_yt" if method == "youtube" else "transcribe_upload"
            page.button(key=action).click().run()
            page.text_area[0].set_value("corrected transcript").run()
            prefix = "yt_result_segment_" if method == "youtube" else "upload_result_segment_"

        assert not page.exception
        assert transcribe.call_count == 1
        downloads = download.call_count
        st.cache_data.clear()
        for index in (1, 1, 0, 2):
            page.button(key=f"{prefix}{index}").click().run()
            assert not page.exception
            assert not page.error
            if method == "search":
                assert "missing" in page.warning[0].value
                assert page.button(key="word_match_1_2").label == "01:00:01"
            else:
                assert page.text_area[0].value == "corrected transcript"
                assert len(page.get("download_button")) == 1
        assert transcribe.call_count == 1
        assert download.call_count == downloads

        if method == "upload":
            upload.seek(0)
            upload.write(b"new recording")
            page.run()
        else:
            page.text_input[0].set_value("https://youtu.be/rdzKbICr1A0").run()
        assert not any(button.key and button.key.startswith(prefix) for button in page.button)
        if method != "search":
            assert not page.text_area
            assert not page.get("download_button")
        assert transcribe.call_count == 1
    st.cache_data.clear()
