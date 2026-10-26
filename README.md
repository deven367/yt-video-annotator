# yt-video-annotator

## Try the app on hf-spaces

You can find the deployed app &rarr; [**here**](https://huggingface.co/spaces/deven367/yt-video-annotator/)

> **Note**
> Transcription runs locally with pywhispercpp. Speed depends on the model size and available hardware.

## Installation

1. Create a Python 3.8+ virtual env with the environment manager of your choice
2. Activate the environment
3. Install [FFmpeg](https://ffmpeg.org/download.html) and ensure `ffmpeg` is on your PATH, then install the Python dependencies using ```pip install -e .```
4. To run the app locally in your terminal, type `run_app` from any writable directory. The installed package includes the main UI and Audio Transcription page; audio and SRT files are stored under the working directory.

Streamlit options can be passed through, for example `run_app --server.port=8502`. Launcher failures return Streamlit's nonzero exit status.

The first transcription downloads the selected Whisper model. Use a YouTube video URL, or open the Audio Transcription page to upload an audio file.

Completed Audio Transcription results and edited-text downloads remain available across page reruns. Changing the video, uploaded recording, or model clears the previous result; ordinary widget interactions do not transcribe again.

Some YouTube videos also require a JavaScript runtime; follow [yt-dlp's setup instructions](https://github.com/yt-dlp/yt-dlp/wiki/EJS) if extraction reports a missing runtime.

To run the regression tests, install the development dependencies with `pip install -e '.[dev]'`, then run `python -m pytest tests`.

## Contributing

Issues and PRs are welcome. If you want me to implement a feature, create a Feature Request in the issues, I'll try my best to implement it.
