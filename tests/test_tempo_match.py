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

module = importlib.import_module("src.tempo_match")
TempoMatch = module.TempoMatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_audio(sample_rate: int = 44100) -> dict:
    waveform = MockTensor(np.random.randn(1, 2, 16000))
    return {"waveform": waveform, "sample_rate": sample_rate}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTempoMatchNode:
    # -- schema tests -------------------------------------------------------

    def test_input_types_has_required_audio_1_and_audio_2(self):
        schema = TempoMatch.INPUT_TYPES()
        required = schema["required"]
        assert "audio_1" in required
        assert required["audio_1"] == ("AUDIO",)
        assert "audio_2" in required
        assert required["audio_2"] == ("AUDIO",)

    def test_return_types(self):
        assert TempoMatch.RETURN_TYPES == ("AUDIO", "AUDIO")

    # -- main logic tests ---------------------------------------------------

    def test_different_tempos_120_and_80(self, monkeypatch):
        """120 and 80 → avg 100, rate_1 = 100/120, rate_2 = 100/80."""
        tempo_values = iter([120.0, 80.0])
        monkeypatch.setattr(module, "estimate_tempo", lambda w, sr: next(tempo_values))

        ts_calls = []

        def fake_time_shift(waveform, rate):
            ts_calls.append(rate)
            return waveform  # pass-through

        monkeypatch.setattr(module, "time_shift", fake_time_shift)

        node = TempoMatch()
        result_1, result_2 = node.main(_make_audio(44100), _make_audio(22050))

        assert ts_calls[0] == pytest.approx(100.0 / 120.0)
        assert ts_calls[1] == pytest.approx(100.0 / 80.0)
        assert result_1["sample_rate"] == 44100
        assert result_2["sample_rate"] == 22050

    def test_same_tempo_rates_are_one(self, monkeypatch):
        monkeypatch.setattr(module, "estimate_tempo", lambda w, sr: 100.0)

        ts_calls = []

        def fake_time_shift(waveform, rate):
            ts_calls.append(rate)
            return waveform

        monkeypatch.setattr(module, "time_shift", fake_time_shift)

        TempoMatch().main(_make_audio(), _make_audio())

        assert ts_calls[0] == pytest.approx(1.0)
        assert ts_calls[1] == pytest.approx(1.0)

    def test_output_waveforms_unsqueezed(self, monkeypatch):
        monkeypatch.setattr(module, "estimate_tempo", lambda w, sr: 100.0)
        monkeypatch.setattr(module, "time_shift", lambda w, r: w)

        result_1, result_2 = TempoMatch().main(_make_audio(), _make_audio())

        assert result_1["waveform"].ndim == 3
        assert result_1["waveform"].shape[0] == 1
        assert result_2["waveform"].ndim == 3
        assert result_2["waveform"].shape[0] == 1

    def test_time_shift_receives_squeezed_waveforms(self, monkeypatch):
        monkeypatch.setattr(module, "estimate_tempo", lambda w, sr: 100.0)

        received_ndims = []

        def spy(waveform, rate):
            received_ndims.append(waveform.ndim)
            return waveform

        monkeypatch.setattr(module, "time_shift", spy)

        TempoMatch().main(_make_audio(), _make_audio())

        assert received_ndims == [2, 2], "both waveforms should be 2-d (batch squeezed)"
