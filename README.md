---
title: Yt Video Annotator
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.66.1
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
4. Run `run_app` from any writable directory. Audio and SRT files are stored under that directory.

Streamlit options can be passed through, for example `run_app --server.port=8502`.

The first transcription downloads the selected Whisper model. Use a YouTube video URL, or open the Audio Transcription page to upload an audio file.

Download, model-loading, and transcription failures are shown as handled errors. Correct the input or backend problem and click the same action to retry; failed operations are not cached as completed transcripts.

Completed Audio Transcription results and edited-text downloads remain available across page reruns. Changing the video, uploaded recording, or model clears the previous result; ordinary widget interactions do not transcribe again.

Click a word-search timestamp or a segment's start time in **View Timestamped Segments** to seek the existing video/audio player. Seeking preserves results, edited text, and downloads without downloading or transcribing again. Clicking the same timestamp seeks again; changing the video, recording, or model resets playback. Media seeks use whole-second precision (segment starts are rounded down); press Play if your browser blocks autoplay.

Some YouTube videos also require a JavaScript runtime; follow [yt-dlp's setup instructions](https://github.com/yt-dlp/yt-dlp/wiki/EJS) if extraction reports a missing runtime.

Audio and transcript caches have bounded retention; evicted videos are downloaded again when needed.

## Contributing

Issues and PRs are welcome. If you want me to implement a feature, create a Feature Request in the issues, I'll try my best to implement it.

See [AGENTS.md](AGENTS.md) for contributor setup, verification commands, playback/cache contracts, and GitHub/Hugging Face deployment instructions.
