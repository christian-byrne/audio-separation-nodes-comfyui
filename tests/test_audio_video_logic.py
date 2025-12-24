import math

import pytest

from src.audio_video_logic import (
    AudioVideoCombineError,
    compute_trim_window,
    parse_timestamp,
)


def test_parse_timestamp_supports_seconds_minutes_and_hours():
    assert math.isclose(parse_timestamp("42", default=0), 42.0)
    assert math.isclose(parse_timestamp("1:30", default=0), 90.0)
    assert math.isclose(parse_timestamp("2:01:05", default=0), 2 * 3600 + 65)


def test_parse_timestamp_requires_default_when_blank():
    with pytest.raises(AudioVideoCombineError):
        parse_timestamp("", default=None)


@pytest.mark.parametrize(
    "start,end,duration,expected",
    [
        ("0:00", "1:00", 90.0, (0.0, 60.0)),
        ("30", "120", 100.0, (30.0, 100.0)),
        ("0", "", 15.0, (0.0, 15.0)),
    ],
)
def test_compute_trim_window_clamps_and_defaults(start, end, duration, expected):
    assert compute_trim_window(start, end, duration) == expected


def test_compute_trim_window_requires_end_time_if_duration_unknown():
    with pytest.raises(AudioVideoCombineError):
        compute_trim_window("0", "", None)


def test_compute_trim_window_validates_start_before_end():
    with pytest.raises(AudioVideoCombineError):
        compute_trim_window("5", "1", 10.0)
