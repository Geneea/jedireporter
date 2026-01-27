from dataclasses import dataclass

from jedireporter.transcript import Transcript


def safe_sum(a: float | None, b: float | None) -> float | None:
    """Returns `None` if both values are `None` otherwise replaces `None` with 0.0"""
    if a is None and b is None:
        return None
    else:
        return (a or 0.0) + (b or 0.0)


@dataclass
class TranscriptPair:
    candidate: Transcript
    source: Transcript
    gold: Transcript | None
