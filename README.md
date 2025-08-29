
# Live Transcription & Translation Toolkit

This repository contains a powerful toolkit for live transcription and automatic translation of audio from various sources, such as YouTube, Twitch, or local video files. It leverages `faster-whisper` for high-performance transcription and `deep-translator` for seamless translation.

The toolkit is offered in two distinct versions to suit different use cases:

1. **`transcription_withoutapi.py`**: A straightforward, command-line interface (CLI) tool for users who want to quickly transcribe a single media source.
2. **`transcription_api.py`**: A robust, full-featured FastAPI server that provides a rich web interface for managing multiple transcription sessions, real-time updates via WebSockets, and a REST API for programmatic control.

## Features

- **Real-Time Transcription**: Processes live streams or growing local files with minimal delay.
- **Automatic Translation**: Automatically translates transcribed text into a target language (e.g., French to English, English to French).
- **High Performance**: Optimized for CPU performance using `faster-whisper` with `int8` computation.
- **Multiple Sources**: Supports any video platform compatible with `streamlink` (YouTube, Twitch, etc.) and local video files.
- **SRT File Generation**: Continuously appends captions to a standard `.srt` subtitle file.
- **User-Friendly Web Interface (API Version)**: Start, stop, and monitor transcription sessions from a web browser. Includes a live video player with synchronized captions.
- **Session Management (API Version)**: Run and manage multiple transcription jobs concurrently.
- **Real-time Updates (API Version)**: Uses WebSockets to push live status and captions to the web UI without needing to refresh.
- **Optional Media Player Integration**: Automatically launch a local media player (like MPC-BE or VLC) on Windows to view the video with live subtitles.

## Prerequisites

Before you begin, ensure you have the following installed:

1. **Python 3.8+**
2. **FFmpeg**: This is required for audio extraction. You can download it from [ffmpeg.org](https://ffmpeg.org/download.html) and ensure it's in your system's PATH.
3. **Streamlink**: For processing live URLs.
   ```bash
   pip install streamlink
   ```
4. **Cloudflared (Optional, for API Server)**: To expose the web server to the internet. Download from the [Cloudflare Zero Trust Dashboard](https://one.dash.cloudflare.com/).

### Python Libraries

Install the required Python packages using pip:

```bash
pip install "faster_whisper" deep-translator ffmpeg-python pysrt tqdm fastapi uvicorn "Jinja2" "python-multipart" "pywinauto; platform_system=='Windows'"
```

## Usage

### 1. Simple CLI Version (`transcription_withoutapi.py`)

This version is ideal for single, command-line-driven transcription tasks.

**To transcribe a live URL (e.g., YouTube):**

```bash
python transcription_withoutapi.py --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --model small --auto-translate
```

**To transcribe a local video file:**

```bash
python transcription_withoutapi.py --file "path/to/your/video.mp4" --model small --auto-translate
```

**CLI Arguments:**

| Argument | Description |
|----------|-------------|
| `--url` | The live stream URL to process. |
| `--file` | The path to a local video file. |
| `--srt` | Path for the output `.srt` file. (Default: `captions.srt`) |
| `--model` | Whisper model size (`tiny`, `small`, `medium`, `large-v3`). (Default: `tiny`) |
| `--auto-translate` | Enable automatic translation. |
| `--play` | (Windows Only) Launch a local media player with the video and subtitles. |
| `--wait` | Interval in seconds to check for new content. (Default: `2`) |

### 2. API Server & Web UI (`transcription_api.py`)

This version provides a comprehensive web interface for managing sessions. It's perfect for more complex workflows or for providing a service to others on your network.

**To start the server:**

```bash
python transcription_api.py
```

Once started, the server will provide several URLs for access:

- **Local Access**: `http://localhost:8000`
- **Network Access**: `http://<your-local-ip>:8000` (accessible by other devices on your Wi-Fi/LAN)
- **Public Access (if Cloudflared is running)**: A `trycloudflare.com` URL.

**Using the Web Interface:**

1. Open one of the provided URLs in your web browser.
2. Enter the media URL you wish to transcribe.
3. Select the desired model size and toggle translation or low-latency mode.
4. Click **"Start Transcription"**.
5. A new session will begin, and you can monitor its progress in the activity log.
6. Once the session is running, click the **"Open Video Player"** link to watch the stream with live, synchronized captions overlaid on the video.

## How It Works

The transcription process follows this pipeline:

1. **Video Stream**: `streamlink` downloads the live video content from the source URL into a local file.
2. **Audio Extraction**: `ffmpeg` polls this file, extracts new audio segments in the required format (16kHz mono WAV), and passes them for processing.
3. **Transcription**: `faster-whisper` takes the audio chunks and rapidly converts speech to text, also detecting the source language.
4. **Translation**: If enabled, `deep-translator` translates the transcribed text into the target language.
5. **Subtitle Generation**: The final captions are formatted and appended to an `.srt` file. For the API server, captions are also broadcast via WebSocket to the web UI in real-time.
