---
title: Yt Video Annotator
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.25.0
python_version: "3.11"
app_file: annotator/app.py
pinned: false
---

# yt-video-annotator

## Try the app on hf-spaces

You can find the deployed app &rarr; [**here**](https://huggingface.co/spaces/deven367/yt-video-annotator-hf/)

> **Note**
> Transcription runs locally with pywhispercpp. Speed depends on the model size and available hardware.

## Installation

1. Create a Python 3.11+ virtual env with the environment manager of your choice.
2. Activate the environment
3. Install [FFmpeg](https://ffmpeg.org/download.html) and ensure `ffmpeg` is on your PATH, then install the Python dependencies using ```pip install -e .```
4. To run the app locally in your terminal, type `run_app` from any writable directory. The installed package includes the main UI and Audio Transcription page; audio and SRT files are stored under the working directory.

Streamlit options can be passed through, for example `run_app --server.port=8502`. Launcher failures return Streamlit's nonzero exit status.

The first transcription downloads the selected Whisper model. Use a YouTube video URL, or open the Audio Transcription page to upload an audio file.

Download, model-loading, and transcription failures are shown as handled errors. Correct the input or backend problem and click the same action to retry; failed operations are not cached as completed transcripts.

Completed Audio Transcription results and edited-text downloads remain available across page reruns. Changing the video, uploaded recording, or model clears the previous result; ordinary widget interactions do not transcribe again.

Some YouTube videos also require a JavaScript runtime; follow [yt-dlp's setup instructions](https://github.com/yt-dlp/yt-dlp/wiki/EJS) if extraction reports a missing runtime.

The main-page transcript cache expires after one hour and holds at most 32 videos. Converted WAVs in `audio/` are limited to 32 files and 512 MiB, keeping the most recently used files. Cleanup runs when audio is acquired and after a conversion is published; evicted videos are downloaded again on demand.

Each operation pins its audio with a temporary hard link, so eviction cannot remove audio still being transcribed. Working paths are removed on context exit, including errors. Oversized audio is processed without retaining a cache entry. Active downloads/processing can temporarily exceed the cache budget; unrelated audio files, models, and SRT outputs are not part of this cache policy.

For Python callers, `get_audio` is a context manager: keep transcription inside `with get_audio(url) as (audio_path, was_cached):` so the working path remains valid.

To run the regression tests, install the development dependencies with `pip install -e '.[dev]'`, then run `python -m pytest tests`.

CI uses uv to build the sdist and wheel on Python 3.11, test the installed wheel outside the checkout, and smoke-check source and editable installations on that minimum version.

## Contributing

Issues and PRs are welcome. If you want me to implement a feature, create a Feature Request in the issues, I'll try my best to implement it.
