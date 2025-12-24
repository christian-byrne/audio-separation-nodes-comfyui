"""Pure helper functions for the AudioVideoCombine node."""

from __future__ import annotations

from typing import Optional, Tuple


class AudioVideoCombineError(ValueError):
    """Custom error to allow targeted unit testing."""


def parse_timestamp(value: str, default: Optional[float]) -> float:
    """Convert ``HH:MM:SS``/``MM:SS`` strings (or seconds) into a float."""

    normalized = (value or "").strip()
    if not normalized:
        if default is None:
            raise AudioVideoCombineError(
                "AudioVideoCombine: A video end time must be provided when the duration cannot be determined."
            )
        return float(default)

    if ":" not in normalized:
        try:
            return float(normalized)
        except ValueError as exc:
            raise AudioVideoCombineError(
                f"AudioVideoCombine: Invalid timestamp '{value}'. Expected MM:SS or HH:MM:SS."
            ) from exc

    parts = normalized.split(":")
    if len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise AudioVideoCombineError(
            f"AudioVideoCombine: Invalid timestamp '{value}'. Expected MM:SS or HH:MM:SS."
        )

    try:
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError as exc:
        raise AudioVideoCombineError(
            f"AudioVideoCombine: Invalid timestamp '{value}'. Expected MM:SS or HH:MM:SS."
        ) from exc


def compute_trim_window(
    start_time: str,
    end_time: str,
    duration_seconds: Optional[float],
) -> Tuple[float, float]:
    """Calculate the numeric trim window for the combine node."""

    start_seconds = parse_timestamp(start_time, default=0.0)
    end_seconds = parse_timestamp(end_time, default=duration_seconds)

    if duration_seconds is not None:
        end_seconds = min(end_seconds, duration_seconds)

    if start_seconds >= end_seconds:
        raise AudioVideoCombineError(
            "AudioVideoCombine: Start time must be less than end time. Start time cannot be after video ends."
        )

    return start_seconds, end_seconds


__all__ = [
    "AudioVideoCombineError",
    "parse_timestamp",
    "compute_trim_window",
]
