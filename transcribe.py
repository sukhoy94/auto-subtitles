import whisper
from datetime import timedelta
import os
import sys
import re

INPUT_VIDEO = "input/video2.MOV"
OUTPUT_SRT = "output/subtitles.srt"

def split_sentences(text: str, max_len=45):
    parts = re.split(r'(,)', text)
    chunks = []
    current = ""

    for part in parts:
        if part == ",":
            current += part
            chunks.append(current.strip())
            current = ""
        else:
            if len(current) + len(part) <= max_len:
                current += part
            else:
                chunks.append(current.strip())
                current = part

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if c]


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



def build_subtitles_from_words(words, pause=0.4, max_len=40):
    subs = []
    current_words = []
    start_time = words[0]["start"]

    for i, w in enumerate(words):
        if current_words:
            gap = w["start"] - current_words[-1]["end"]
        else:
            gap = 0

        text_len = len(" ".join(x["word"] for x in current_words))

        if gap > pause or text_len > max_len:
            end_time = current_words[-1]["end"]
            subs.append((start_time, end_time, " ".join(x["word"] for x in current_words)))
            current_words = []
            start_time = w["start"]

        current_words.append(w)

    if current_words:
        subs.append((start_time, current_words[-1]["end"], " ".join(x["word"] for x in current_words)))

    return subs


def main():
    validate_input(INPUT_VIDEO)

    model = whisper.load_model("medium")
    result = model.transcribe(
        INPUT_VIDEO,
        language="pl",
        word_timestamps=True
    )

    os.makedirs("output", exist_ok=True)


    with open(OUTPUT_SRT, "w", encoding="utf-8") as f:
        index = 1

        for segment in result["segments"]:
            if "words" not in segment or not segment["words"]:
                continue

            subtitles = build_subtitles_from_words(segment["words"])

            for start, end, text in subtitles:
                f.write(f"{index}\n")
                f.write(f"{format_time(start)} --> {format_time(end)}\n")
                f.write(f"{text.strip()}\n\n")
                index += 1


if __name__ == "__main__":
    main()
