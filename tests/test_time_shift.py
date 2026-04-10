from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock heavy dependencies before importing the module under test
# ---------------------------------------------------------------------------

torch_mock = types.ModuleType("torch")


class MockTensor:
    def __init__(self, data):
        self._data = np.array(data) if not isinstance(data, np.ndarray) else data

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    def squeeze(self, dim=0):
        return MockTensor(np.squeeze(self._data, axis=dim))

    def unsqueeze(self, dim=0):
        return MockTensor(np.expand_dims(self._data, axis=dim))

    def float(self):
        return self

    def to(self, device):
        return self

    def numpy(self):
        return self._data


torch_mock.Tensor = MockTensor
torch_mock.device = str
torch_mock.hann_window = lambda *a, **kw: MockTensor(np.ones(2048))
torch_mock.stft = lambda *a, **kw: MockTensor(np.zeros((2, 1025, 10)))
torch_mock.istft = lambda *a, **kw: MockTensor(np.zeros((2, 16000)))
torch_mock.linspace = lambda *a, **kw: MockTensor(np.zeros((1025, 1)))
torch_mock.cfloat = "torch.cfloat"
torch_mock.no_grad = type("_NoGrad", (), {"__enter__": lambda s: s, "__exit__": lambda s, *a: None})
sys.modules["torch"] = torch_mock

torchaudio_mock = types.ModuleType("torchaudio")
torchaudio_functional = types.ModuleType("torchaudio.functional")
torchaudio_functional.phase_vocoder = lambda *a, **kw: MockTensor(np.zeros((2, 1025, 10)))
torchaudio_mock.functional = torchaudio_functional
sys.modules["torchaudio"] = torchaudio_mock
sys.modules["torchaudio.functional"] = torchaudio_functional

librosa_mock = types.ModuleType("librosa")
librosa_onset = types.ModuleType("librosa.onset")
librosa_beat = types.ModuleType("librosa.beat")
librosa_mock.onset = librosa_onset
librosa_mock.beat = librosa_beat
sys.modules["librosa"] = librosa_mock
sys.modules["librosa.onset"] = librosa_onset
sys.modules["librosa.beat"] = librosa_beat

# Clear any previously-cached src modules so they reimport with our mocks
for _key in list(sys.modules):
    if _key.startswith("src."):
        del sys.modules[_key]

# ---------------------------------------------------------------------------
# Import module under test
# ---------------------------------------------------------------------------

module = importlib.import_module("src.time_shift")
TimeShift = module.TimeShift


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_audio(sample_rate: int = 44100) -> dict:
    waveform = MockTensor(np.random.randn(1, 2, 16000))
    return {"waveform": waveform, "sample_rate": sample_rate}


def _stub_time_shift(returned_data=None):
    """Return a (spy, mock_result) pair for time_shift."""
    mock_result = MockTensor(returned_data if returned_data is not None else np.random.randn(2, 8000))
    calls = []

    def spy(waveform, rate):
        calls.append({"waveform_ndim": waveform.ndim, "rate": rate})
        return mock_result

    return spy, calls, mock_result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTimeShiftNode:
    # -- schema tests -------------------------------------------------------

    def test_input_types_has_required_audio_and_rate(self):
        schema = TimeShift.INPUT_TYPES()
        required = schema["required"]
        assert "audio" in required
        assert required["audio"] == ("AUDIO",)
        assert "rate" in required

    def test_rate_constraints(self):
        schema = TimeShift.INPUT_TYPES()
        rate_spec = schema["required"]["rate"]
        assert rate_spec[0] == "FLOAT"
        opts = rate_spec[1]
        assert opts["default"] == 1.0
        assert opts["min"] == 0.1
        assert opts["max"] == 10.0

    def test_return_types(self):
        assert TimeShift.RETURN_TYPES == ("AUDIO",)

    # -- main logic tests ---------------------------------------------------

    def test_basic_call(self, monkeypatch):
        spy, calls, mock_result = _stub_time_shift()
        monkeypatch.setattr(module, "time_shift", spy)

        node = TimeShift()
        (result,) = node.main(_make_audio(22050), rate=1.5)

        assert calls[0]["rate"] == 1.5
        assert result["sample_rate"] == 22050

    def test_rate_clamped_below_min(self, monkeypatch):
        spy, calls, _ = _stub_time_shift()
        monkeypatch.setattr(module, "time_shift", spy)

        TimeShift().main(_make_audio(), rate=0.05)
        assert calls[0]["rate"] == pytest.approx(0.1)

    def test_rate_clamped_above_max(self, monkeypatch):
        spy, calls, _ = _stub_time_shift()
        monkeypatch.setattr(module, "time_shift", spy)

        TimeShift().main(_make_audio(), rate=15.0)
        assert calls[0]["rate"] == pytest.approx(10.0)

    def test_rate_within_range_unchanged(self, monkeypatch):
        spy, calls, _ = _stub_time_shift()
        monkeypatch.setattr(module, "time_shift", spy)

        TimeShift().main(_make_audio(), rate=1.5)
        assert calls[0]["rate"] == pytest.approx(1.5)

    def test_waveform_squeezed_before_call(self, monkeypatch):
        spy, calls, _ = _stub_time_shift()
        monkeypatch.setattr(module, "time_shift", spy)

        TimeShift().main(_make_audio(), rate=1.0)
        assert calls[0]["waveform_ndim"] == 2, "batch dim should be squeezed"

    def test_output_waveform_unsqueezed(self, monkeypatch):
        data_2d = np.random.randn(2, 8000)
        spy, _, _ = _stub_time_shift(data_2d)
        monkeypatch.setattr(module, "time_shift", spy)

        (result,) = TimeShift().main(_make_audio(), rate=1.0)
        assert result["waveform"].ndim == 3, "output should have batch dim restored"
        assert result["waveform"].shape[0] == 1
