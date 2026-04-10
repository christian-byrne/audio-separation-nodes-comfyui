from __future__ import annotations

import importlib
import sys
import types

import numpy as np

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

module = importlib.import_module("src.get_tempo")
GetTempo = module.GetTempo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_audio(sample_rate: int = 44100) -> dict:
    """Return a minimal AUDIO dict with a batch-dim waveform."""
    waveform = MockTensor(np.random.randn(1, 2, 16000))
    return {"waveform": waveform, "sample_rate": sample_rate}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetTempoNode:
    def test_input_types_has_required_audio(self):
        schema = GetTempo.INPUT_TYPES()
        assert "required" in schema
        assert "audio" in schema["required"]
        assert schema["required"]["audio"] == ("AUDIO",)

    def test_return_types(self):
        assert GetTempo.RETURN_TYPES == ("STRING", "FLOAT", "INTEGER")

    def test_return_names(self):
        assert GetTempo.RETURN_NAMES == ("tempo_string", "tempo_float", "tempo_integer")

    def test_tempo_120_7(self, monkeypatch):
        monkeypatch.setattr(module, "estimate_tempo", lambda w, sr: 120.7)
        node = GetTempo()
        result = node.main(_make_audio())
        assert result == ("121", 120.7, 120)

    def test_tempo_85_3(self, monkeypatch):
        monkeypatch.setattr(module, "estimate_tempo", lambda w, sr: 85.3)
        node = GetTempo()
        result = node.main(_make_audio())
        assert result == ("85", 85.3, 85)

    def test_waveform_is_squeezed_before_call(self, monkeypatch):
        """estimate_tempo should receive a 2-d tensor (batch dim squeezed)."""
        received = {}

        def spy(waveform, sample_rate):
            received["ndim"] = waveform.ndim
            received["shape"] = waveform.shape
            return 100.0

        monkeypatch.setattr(module, "estimate_tempo", spy)
        node = GetTempo()
        node.main(_make_audio(44100))

        assert received["ndim"] == 2, "waveform should be squeezed to 2-d"
