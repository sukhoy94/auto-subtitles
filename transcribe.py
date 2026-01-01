import whisper
from datetime import timedelta
import os
import sys
import re
import argparse

DEFAULTS = {
    "input": "input/video2.MOV",
    "output": "output/subtitles.srt",
    "model": "medium",
    "language": "en",  # Changed default to English
    "pause": 0.4,
    "max_len": 40,
}

def ask(prompt: str, default, cast=str):
    value = input(f"{prompt} [{default}]: ").strip()
    if value == "":
        return default
    try:
        return cast(value)
    except ValueError:
        print("Invalid value, using default.")
        return default


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SRT subtitles using OpenAI Whisper"
    )

    parser.add_argument("--input", help="Path to input video file")
    parser.add_argument("--output", help="Path to output SRT file")
    parser.add_argument("--model", choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--language", help="Language code (e.g., en, pl, es)")
    parser.add_argument("--pause", type=float, help="Max pause between words in seconds")
    parser.add_argument("--max-len", type=int, help="Max characters per subtitle line")

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Ask for missing parameters interactively"
    )

    return parser.parse_args()


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
        print("👉 Make sure the video file exists in the specified directory.")
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
    args = parse_args()

    if args.interactive:
        args.input = args.input or ask(
            "Path to video file", DEFAULTS["input"]
        )
        args.output = args.output or ask(
            "Path to output SRT file", DEFAULTS["output"]
        )
        args.model = args.model or ask(
            "Whisper model (tiny/base/small/medium/large)",
            DEFAULTS["model"]
        )
        args.language = args.language or ask(
            "Language code",
            DEFAULTS["language"]
        )
        args.pause = args.pause or ask(
            "Pause between words (seconds)",
            DEFAULTS["pause"],
            float
        )
        args.max_len = args.max_len or ask(
            "Maximum subtitle length",
            DEFAULTS["max_len"],
            int
        )
    else:
        # Fallback to defaults
        for key, value in DEFAULTS.items():
            if getattr(args, key) is None:
                setattr(args, key, value)

    validate_input(args.input)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model: {args.model}...")
    model = whisper.load_model(args.model)
    
    print(f"Transcribing file: {args.input}...")
    result = model.transcribe(
        args.input,
        language=args.language,
        word_timestamps=True
    )

    with open(args.output, "w", encoding="utf-8") as f:
        index = 1

        for segment in result["segments"]:
            if "words" not in segment or not segment["words"]:
                continue

            subtitles = build_subtitles_from_words(
                segment["words"],
                pause=args.pause,
                max_len=args.max_len
            )

            for start, end, text in subtitles:
                f.write(f"{index}\n")
                f.write(f"{format_time(start)} --> {format_time(end)}\n")
                f.write(f"{text.strip()}\n\n")
                index += 1
    
    print(f"Done! Subtitles saved to: {args.output}")


if __name__ == "__main__":
    main()
    