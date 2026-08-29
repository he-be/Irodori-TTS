"""Split a long script into segments short enough to synthesize one at a time.

The runtime generates a whole utterance in one shot, and both cost and quality get
worse with length: the DiT spends 73 / 79 / 136 ms per second of audio for 7.2 /
11.8 / 28.8 s outputs, and outputs past 20 s are pushed to 16 sampler steps
(docs/experiments/14-step-count.md, 15-decode-ane.md 5-1/5-4). Segments of roughly
7-12 s are the sweet spot, so the default budget is 60 characters
(~12 s at the measured rate below).

Splitting is text-only: sentence boundaries first, then clause punctuation for a
sentence that is too long on its own, and a hard cut as the last resort.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 4.9 characters per second, measured on the benchmark inputs (59 chars -> 11.8 s,
# 140 chars -> 28.8 s; docs/experiments/15-decode-ane.md 5-1). Only used to show an
# estimate in the UI and to pick the default budget - the runtime's duration
# predictor decides the real length.
CHARS_PER_SECOND = 4.9

DEFAULT_MAX_CHARS = 60
DEFAULT_MIN_CHARS = 12
# A merge is allowed to overshoot the budget by this factor, to avoid leaving a
# two-word segment behind on its own - but never past HARD_MAX_CHARS (~22 s), which
# keeps every segment clear of the runtime's 30 s ceiling and of the 20 s point where
# the sampler switches to 16 steps.
_MERGE_SLACK = 1.3
HARD_MAX_CHARS = 110

_SENTENCE_END = "。!?！？…"
# Punctuation that belongs to the sentence that just ended, not to the next one.
_CLOSERS = "」』）)】〉》”’\"'～ー"
_CLAUSE_END = "、,，;；:："

_PARAGRAPH_SPLIT = re.compile(r"\n[ \t　]*\n[\s　]*")


@dataclass(frozen=True)
class Segment:
    text: str
    # True when a blank line separated this segment from the previous one; the app
    # turns that into a longer pause when it concatenates.
    paragraph_break: bool = False

    @property
    def estimated_seconds(self) -> float:
        return len(self.text) / CHARS_PER_SECOND


def _join(left: str, right: str) -> str:
    """Concatenate two pieces, restoring the word space Latin text needs."""
    if not left:
        return right
    if left[-1].isascii() and left[-1].isalnum() and right[:1].isascii() and right[:1].isalnum():
        return f"{left} {right}"
    return left + right


def _cut_after(text: str, enders: str) -> list[str]:
    """Cut after every run of `enders` (plus any closing brackets that follow it)."""
    pieces: list[str] = []
    start = 0
    i = 0
    while i < len(text):
        if text[i] in enders:
            while i + 1 < len(text) and text[i + 1] in enders:
                i += 1
            end = i + 1
            while end < len(text) and text[end] in _CLOSERS:
                end += 1
            piece = text[start:end].strip()
            if piece:
                pieces.append(piece)
            start = end
            i = end
            continue
        i += 1
    tail = text[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces


def _fit(text: str, max_chars: int) -> list[str]:
    """Break one sentence that exceeds the budget into pieces that do not."""
    if len(text) <= max_chars:
        return [text]

    out: list[str] = []
    for clause in _cut_after(text, _CLAUSE_END):
        if len(clause) <= max_chars:
            out.append(clause)
            continue
        # No usable punctuation left: pack whole words, and hard-cut only a single
        # word (or an unbroken CJK run) that is longer than the budget by itself.
        line = ""
        for word in re.split(r"[ 　]+", clause):
            if not word:
                continue
            candidate = _join(line, word)
            if line and len(candidate) > max_chars:
                out.append(line)
                line = word
            else:
                line = candidate
            while len(line) > max_chars:
                out.append(line[:max_chars])
                line = line[max_chars:]
        if line:
            out.append(line)
    return out or [text]


def _pack(pieces: list[str], max_chars: int, min_chars: int) -> list[str]:
    packed: list[str] = []
    for piece in pieces:
        if packed and len(_join(packed[-1], piece)) <= max_chars:
            packed[-1] = _join(packed[-1], piece)
        else:
            packed.append(piece)

    # A leftover shorter than min_chars reads as a fragment on its own; fold it into
    # a neighbour when the result stays close to the budget.
    limit = min(int(max_chars * _MERGE_SLACK), HARD_MAX_CHARS)
    merged: list[str] = []
    for piece in packed:
        if merged and len(piece) < min_chars and len(merged[-1]) + len(piece) <= limit:
            merged[-1] = _join(merged[-1], piece)
        else:
            merged.append(piece)
    if len(merged) >= 2 and len(merged[0]) < min_chars and len(merged[0]) + len(merged[1]) <= limit:
        merged[1] = _join(merged[0], merged[1])
        merged.pop(0)
    return merged


def split_script(
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[Segment]:
    """Split `text` into segments of at most `max_chars` characters.

    Blank lines always start a new segment and are reported as `paragraph_break`.
    Single line breaks are sentence boundaries but may be packed together.
    """
    if max_chars < 8:
        raise ValueError(f"max_chars must be >= 8, got {max_chars}")
    min_chars = max(0, min(int(min_chars), max_chars))

    segments: list[Segment] = []
    for paragraph in _PARAGRAPH_SPLIT.split(str(text).strip()):
        pieces: list[str] = []
        for line in paragraph.splitlines():
            line = line.strip()
            if not line:
                continue
            for sentence in _cut_after(line, _SENTENCE_END):
                pieces.extend(_fit(sentence, max_chars))
        for index, packed in enumerate(_pack(pieces, max_chars, min_chars)):
            segments.append(Segment(text=packed, paragraph_break=bool(index == 0 and segments)))
    return segments
