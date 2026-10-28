"""Transcribe an interview recording on this computer (nothing is uploaded).

Setup (once, in the musiccrs environment):
    pip install faster-whisper

Usage, from the repo root:
    python -m scripts.transcribe interview.m4a
    python -m scripts.transcribe interview.m4a --language en --model medium
    python -m scripts.transcribe interview.m4a --redact "Anna,Ola Nordmann"

Writes next to the audio file:
    interview.transcript.txt   one line per spoken segment: [mm:ss] text
    interview.srt              the same with subtitle timing (handy to check against the audio)

The first run downloads the speech model (small: about 0.5 GB). The text has no speaker labels;
add "Interviewer:" / "Participant:" by hand and read it once for misheard words.
"""

import argparse
import re
import time
from pathlib import Path

from faster_whisper import WhisperModel


def clock(seconds: float, hours: bool = False) -> str:
    s = int(seconds)
    h, m, s = s // 3600, s // 60 % 60, s % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if hours else f"{h * 60 + m:02d}:{s:02d}"


def srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def redactor(names: str):
    """Replace each comma-separated name (whole words, any case) with [NAME]."""
    words = [n.strip() for n in names.split(",") if n.strip()]
    if not words:
        return lambda text: text
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b", re.I)
    return lambda text: pattern.sub("[NAME]", text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe an audio file with Whisper (local).")
    parser.add_argument("audio", type=Path, help="m4a, mp3, wav, mp4 ... (anything ffmpeg-style tools read)")
    parser.add_argument("--model", default="small", help="tiny, base, small, medium, large-v3 (bigger = slower, better)")
    parser.add_argument("--language", default=None, help="en, no, ... (default: detect from the audio)")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: next to the audio)")
    parser.add_argument("--redact", default="", help="comma-separated names to replace with [NAME]")
    args = parser.parse_args()

    if not args.audio.is_file():
        raise SystemExit(f"File not found: {args.audio}")
    out_dir = args.out or args.audio.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / args.audio.stem
    redact = redactor(args.redact)

    print(f"Loading model '{args.model}' (downloads on first use)...")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    print(f"Transcribing {args.audio.name} ...")
    start = time.time()
    segments, info = model.transcribe(
        str(args.audio),
        language=args.language,
        beam_size=5,
        vad_filter=True,                  # skip long silences and background noise
        condition_on_previous_text=False,  # avoids repeated sentences after a mistake
    )
    print(f"Language: {info.language} (probability {info.language_probability:.2f}), "
          f"length {clock(info.duration, hours=True)}")

    lines, srt = [], []
    for seg in segments:
        text = redact(seg.text.strip())
        if not text:
            continue
        lines.append(f"[{clock(seg.start)}] {text}")
        srt.append(f"{len(srt) + 1}\n{srt_time(seg.start)} --> {srt_time(seg.end)}\n{text}\n")
        print(f"  [{clock(seg.start)}] {text}  ({(time.time() - start):.0f} s elapsed)")

    Path(f"{stem}.transcript.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(f"{stem}.srt").write_text("\n".join(srt), encoding="utf-8")
    print(f"\nDone in {time.time() - start:.0f} s: {len(lines)} segments")
    print(f"  {Path(f'{stem}.transcript.txt')}")
    print(f"  {Path(f'{stem}.srt')}")


if __name__ == "__main__":
    main()
