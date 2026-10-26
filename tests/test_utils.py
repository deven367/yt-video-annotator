import os
import shutil
import subprocess
import wave
from pathlib import Path
from unittest.mock import patch

import pytest
from pywhispercpp.model import Segment

from annotator import utils

VIDEO_ID = "C0DPdy98e4c"
URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


@pytest.mark.parametrize(
    "url",
    [
        URL,
        f"https://youtu.be/{VIDEO_ID}?si=shared",
        f"https://www.youtube.com/watch?feature=shared&v={VIDEO_ID}",
        f"https://www.youtube.com/shorts/{VIDEO_ID}",
        f"https://www.youtube.com/live/{VIDEO_ID}",
    ],
)
def test_video_urls(url):
    assert utils.get_v_from_url(url) == VIDEO_ID


@pytest.mark.parametrize(
    "url",
    [
        "",
        "https://youtube.com/watch",
        "https://youtu.be/../../app.py",
        f"https://example.com/watch?v={VIDEO_ID}",
        f"https://youtube.com.evil.example/watch?v={VIDEO_ID}",
    ],
)
def test_invalid_urls_are_rejected(url):
    with pytest.raises(ValueError):
        utils.get_v_from_url(url)


def test_timestamps_search_and_subtitles():
    df = utils.df_from_result(
        [
            Segment(0, 125, "hello world"),
            Segment(125, 250, " hello again"),
            Segment(250, 360000, " goodbye"),
        ]
    )
    assert df["start"].tolist() == [
        "00:00:00.000",
        "00:00:01.250",
        "00:00:02.500",
    ]
    assert df["end"].iloc[-1] == "01:00:00.000"
    assert utils.find_word_timestamp(df, "hello", "world", "missing") == [
        ["00:00:00", "00:00:01"],
        ["00:00:00"],
        [],
    ]
    assert utils.generate_srt(df).startswith(
        "1\n00:00:00,000 --> 00:00:01,250\nhello world\n\n"
    )


def test_empty_transcription():
    df = utils.df_from_result([])
    assert df.columns.tolist() == ["start", "end", "text"]
    assert utils.generate_srt(df) == ""
    assert utils.find_word_timestamp(df, "hello") == [[]]


def test_blank_audio_is_filtered():
    speech = Segment(0, 100, "hello")
    assert utils.filter_blank_audio([speech, Segment(100, 200, "[BLANK_AUDIO]")]) == [speech]


def test_failed_conversion_preserves_existing_audio(tmp_path):
    output = tmp_path / "converted.wav"
    output.write_bytes(b"previous valid audio")

    def fail(command, **kwargs):
        Path(command[-1]).write_bytes(b"partial output")
        raise subprocess.CalledProcessError(1, command)

    with patch.object(utils.subprocess, "run", side_effect=fail):
        assert not utils.convert_to_whisper_format(tmp_path / "source.wav", output)
    assert output.read_bytes() == b"previous valid audio"
    assert list(tmp_path.iterdir()) == [output]


def test_failed_conversion_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "AUDIO_PATH", tmp_path)
    output = tmp_path / f"{VIDEO_ID}_converted.wav"
    with patch.object(
        utils.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "ffmpeg")
    ):
        assert not utils.convert_to_whisper_format(tmp_path / "source.wav", output)
    assert utils.get_cached_audio(VIDEO_ID) is None


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_real_ffmpeg_conversion(tmp_path):
    source = tmp_path / "stereo.wav"
    output = tmp_path / "converted.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setparams((2, 2, 44100, 4410, "NONE", "not compressed"))
        audio.writeframes(b"\0" * 4410 * 4)
    assert utils.convert_to_whisper_format(source, output)
    with wave.open(str(output), "rb") as audio:
        assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (
            1,
            2,
            16000,
        )
        assert audio.getnframes() > 0


def test_download_and_cache(tmp_path, monkeypatch):
    audio_dir = tmp_path / "audio"
    monkeypatch.setattr(utils, "AUDIO_PATH", audio_dir)
    original_cwd = Path.cwd()

    def download(command, **kwargs):
        assert Path.cwd() == original_cwd
        assert kwargs["check"] is True
        assert "--no-playlist" in command
        output = command[command.index("-o") + 1].replace("%(ext)s", "wav")
        Path(output).write_bytes(b"downloaded audio")

    def convert(source, output):
        assert source.read_bytes() == b"downloaded audio"
        output.write_bytes(b"converted audio")
        return True

    with patch.object(utils.subprocess, "run", side_effect=download) as run:
        with patch.object(utils, "convert_to_whisper_format", side_effect=convert):
            with utils.get_audio(URL) as (audio, cached):
                assert not cached
                assert audio.read_bytes() == b"converted audio"
                with utils.get_audio(URL) as (reused, cached):
                    assert cached
                    assert reused.read_bytes() == audio.read_bytes()
    assert run.call_count == 1
    assert not audio.exists()
    assert list(audio_dir.iterdir()) == [utils.get_cached_audio(VIDEO_ID)]


@pytest.mark.parametrize("failure", ["download", "conversion"])
def test_download_failures_do_not_return_stale_or_partial_audio(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setattr(utils, "AUDIO_PATH", tmp_path)
    cached = tmp_path / f"{VIDEO_ID}_converted.wav"
    cached.write_bytes(b"previous valid audio")

    def download(command, **kwargs):
        if failure == "download":
            raise subprocess.CalledProcessError(1, command)
        output = command[command.index("-o") + 1].replace("%(ext)s", "wav")
        Path(output).write_bytes(b"downloaded audio")

    with patch.object(utils.subprocess, "run", side_effect=download):
        with patch.object(utils, "convert_to_whisper_format", return_value=False):
            with pytest.raises((subprocess.CalledProcessError, RuntimeError)):
                with utils.get_audio(URL, force_download=True):
                    pytest.fail("Failed processing must not yield an audio path")
    assert cached.read_bytes() == b"previous valid audio"
    assert list(tmp_path.iterdir()) == [cached]


@pytest.mark.parametrize("bound", ["bytes", "entries"])
def test_eviction_preserves_active_audio_and_allows_redownload(tmp_path, monkeypatch, bound):
    monkeypatch.setattr(utils, "AUDIO_PATH", tmp_path)
    if bound == "bytes":
        monkeypatch.setattr(utils, "AUDIO_CACHE_MAX_BYTES", 5)
    else:
        monkeypatch.setattr(utils, "AUDIO_CACHE_MAX_ENTRIES", 1)
    original = tmp_path / f"{VIDEO_ID}_converted.wav"
    original.write_bytes(b"older")
    unrelated = tmp_path / "recording.wav"
    unrelated.write_bytes(b"keep me")
    other_url = "https://youtu.be/rdzKbICr1A0"

    def download(command, **kwargs):
        output = command[command.index("-o") + 1].replace("%(ext)s", "wav")
        Path(output).write_bytes(b"downloaded audio")

    def convert(source, output):
        output.write_bytes(b"fresh")
        return True

    with (
        patch.object(utils.subprocess, "run", side_effect=download) as run,
        patch.object(utils, "convert_to_whisper_format", side_effect=convert),
    ):
        with utils.get_audio(URL) as (active, cached):
            assert cached
            os.utime(active, (1, 1))
            with utils.get_audio(other_url) as (other, cached):
                assert not cached
                assert utils.get_cached_audio(VIDEO_ID) is None
                assert active.read_bytes() == b"older"
                with utils.get_audio(other_url) as (reused, cached):
                    assert cached
                    assert reused.read_bytes() == b"fresh"
                os.utime(other, (1, 1))
                with utils.get_audio(URL) as (downloaded, cached):
                    assert not cached
                    assert downloaded.read_bytes() == b"fresh"
                    assert active.read_bytes() == b"older"
                    assert other.read_bytes() == b"fresh"
        assert run.call_count == 2
    assert not active.exists()
    assert not other.exists()
    assert set(tmp_path.iterdir()) == {original, unrelated}
    assert unrelated.read_bytes() == b"keep me"


def test_oversized_audio_is_only_retained_during_processing(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "AUDIO_PATH", tmp_path)
    monkeypatch.setattr(utils, "AUDIO_CACHE_MAX_BYTES", 4)
    cached_path = tmp_path / f"{VIDEO_ID}_converted.wav"

    def download(command, **kwargs):
        Path(command[command.index("-o") + 1].replace("%(ext)s", "wav")).write_bytes(b"source")

    def convert(source, output):
        output.write_bytes(b"large")
        return True

    with (
        patch.object(utils.subprocess, "run", side_effect=download),
        patch.object(utils, "convert_to_whisper_format", side_effect=convert),
    ):
        with pytest.raises(RuntimeError, match="inference failed"):
            with utils.get_audio(URL) as (audio, was_cached):
                assert not was_cached
                assert not cached_path.exists()
                assert audio.read_bytes() == b"large"
                raise RuntimeError("inference failed")
    assert not audio.exists()
    assert not list(tmp_path.iterdir())
