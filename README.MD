# auto-subtitles

A simple Python tool that generates SRT subtitles from video files using OpenAI Whisper. Perfect for Final Cut Pro, Shorts, and other social media videos.

---

## Features

- Generates `.srt` subtitles automatically from video
- Works with `.mp4`, `.mov`, `.mkv`, and other formats supported by ffmpeg
- User-friendly messages when input files are missing
- Ready to import into Final Cut Pro
- Easily extendable for short-form videos

---

## Requirements

- Python 3.10+
- ffmpeg installed and in PATH
- pip

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/auto-subtitles.git
cd auto-subtitles
```

Create venv:

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS / Linux
# .venv\Scripts\activate   # Windows
```

Install dependencies:

```bash
pip install --upgrade pip
pip install openai-whisper ffmpeg-python
```

## Usage
1. Put your video file in the input/ folder, e.g. input/video.mp4.

2. Run the script:

```
python transcribe.py
```

3. Subtitles will be generated in output/subtitles.srt.