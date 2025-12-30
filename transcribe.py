import whisper
from datetime import timedelta
import os
import sys

INPUT_VIDEO = "input/video.MOV"
OUTPUT_SRT = "output/subtitles.srt"


def format_time(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    millis = int((seconds - total_seconds) * 1000)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def validate_input(path: str) -> None:
    if not os.path.exists(path):
        print(f"Input file not found: {path}")
        print("👉 Make sure the video file exists in the 'input/' directory")
        sys.exit(1)

    if not os.path.isfile(path):
        print(f"Input path is not a file: {path}")
        sys.exit(1)



def main():
    validate_input(INPUT_VIDEO)

    model = whisper.load_model("medium")
    result = model.transcribe(INPUT_VIDEO, language="pl")

    os.makedirs("output", exist_ok=True)

    with open(OUTPUT_SRT, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"], start=1):
            start = format_time(segment["start"])
            end = format_time(segment["end"])
            text = segment["text"].strip()

            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")

    print(f"ubtitles generated successfully: {OUTPUT_SRT}")


if __name__ == "__main__":
    main()
