from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock torch before importing the module under test
# ---------------------------------------------------------------------------


class MockTensor:
    """Numpy-backed tensor supporting .shape and [..., start:end] slicing."""

    def __init__(self, data: np.ndarray):
        self._data = data

    @property
    def shape(self):
        return self._data.shape

    def __getitem__(self, key):
        return MockTensor(self._data[key])

    def __eq__(self, other):
        if isinstance(other, MockTensor):
            return np.array_equal(self._data, other._data)
        return NotImplemented


if "torch" not in sys.modules:
    _torch = types.SimpleNamespace(Tensor=object)
    sys.modules["torch"] = _torch

module = importlib.import_module("src.crop")
AudioCrop = module.AudioCrop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_audio(num_frames: int, sample_rate: int = 100):
    waveform = MockTensor(np.arange(num_frames, dtype=np.float32).reshape(1, 1, -1))
    return {"waveform": waveform, "sample_rate": sample_rate}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAudioCrop:
    def test_basic_crop(self):
        audio = _make_audio(5000, sample_rate=100)
        (result,) = AudioCrop().main(audio, start_time="0:10", end_time="0:30")

        assert result["sample_rate"] == 100
        assert result["waveform"].shape == (1, 1, 2000)

    def test_seconds_only_input(self):
        audio = _make_audio(5000, sample_rate=100)
        (result,) = AudioCrop().main(audio, start_time="10", end_time="30")

        assert result["waveform"].shape == (1, 1, 2000)

    def test_start_beyond_audio_length(self):
        audio = _make_audio(500, sample_rate=100)
        # start = 0:10 → frame 1000, but audio only has 500 frames → clamped to 499
        # end   = 0:30 → frame 3000 → clamped to 499
        # start (499) == end (499) → empty slice
        (result,) = AudioCrop().main(audio, start_time="0:10", end_time="0:30")
        assert result["waveform"].shape[-1] == 0

    def test_end_beyond_audio_length(self):
        audio = _make_audio(2000, sample_rate=100)
        # start = 0:05 → frame 500
        # end   = 0:30 → frame 3000 → clamped to 1999
        (result,) = AudioCrop().main(audio, start_time="0:05", end_time="0:30")
        assert result["waveform"].shape == (1, 1, 1499)

    def test_start_greater_than_end_raises(self):
        audio = _make_audio(5000, sample_rate=100)
        with pytest.raises(ValueError, match="Start time must be less than end time"):
            AudioCrop().main(audio, start_time="0:30", end_time="0:10")

    def test_full_duration_crop(self):
        audio = _make_audio(3000, sample_rate=100)
        # end = 0:30 → frame 3000 → clamped to 2999
        (result,) = AudioCrop().main(audio, start_time="0:00", end_time="0:30")
        assert result["waveform"].shape == (1, 1, 2999)

    def test_input_types_schema(self):
        schema = AudioCrop.INPUT_TYPES()
        required = schema["required"]
        assert "audio" in required
        assert "start_time" in required
        assert "end_time" in required

    def test_return_types(self):
        assert AudioCrop.RETURN_TYPES == ("AUDIO",)
