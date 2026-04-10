import math
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock torch, torchaudio, and librosa BEFORE importing src.utils
# ---------------------------------------------------------------------------


class MockTensor:
    """Wraps a numpy array to emulate torch.Tensor for ensure_stereo tests."""

    def __init__(self, data):
        self._data = np.array(data, dtype=np.float32) if not isinstance(data, np.ndarray) else data
        self._dtype = "torch.cfloat"

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def shape(self):
        return self._data.shape

    @property
    def device(self):
        return "cpu"

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = value

    def repeat(self, *sizes):
        return MockTensor(np.tile(self._data, sizes))

    def narrow(self, dim, start, length):
        slices = [slice(None)] * self._data.ndim
        slices[dim] = slice(start, start + length)
        return MockTensor(self._data[tuple(slices)])

    def mean(self, dim=None, keepdim=False):
        return MockTensor(np.mean(self._data, axis=dim, keepdims=keepdim))

    def __getitem__(self, key):
        return MockTensor(self._data[key])

    def dim(self):
        return self._data.ndim

    def squeeze(self, dim):
        return MockTensor(np.squeeze(self._data, axis=dim))

    def numpy(self):
        return self._data

    def __eq__(self, other):
        if isinstance(other, MockTensor):
            return np.array_equal(self._data, other._data)
        return NotImplemented


# -- torch mock -------------------------------------------------------------
torch_mock = types.ModuleType("torch")
torch_mock.Tensor = MockTensor
torch_mock.cfloat = "torch.cfloat"
torch_mock.hann_window = MagicMock(return_value=MockTensor(np.ones(2048)))
torch_mock.stft = MagicMock()
torch_mock.istft = MagicMock()
torch_mock.linspace = MagicMock(return_value=MockTensor(np.zeros((1025, 1))))


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


torch_mock.no_grad = _NoGrad
sys.modules["torch"] = torch_mock

# -- torchaudio mock --------------------------------------------------------
torchaudio_mock = types.ModuleType("torchaudio")
torchaudio_functional_mock = types.ModuleType("torchaudio.functional")
torchaudio_functional_mock.phase_vocoder = MagicMock()
torchaudio_mock.functional = torchaudio_functional_mock
sys.modules["torchaudio"] = torchaudio_mock
sys.modules["torchaudio.functional"] = torchaudio_functional_mock

# -- librosa mock -----------------------------------------------------------
librosa_mock = types.ModuleType("librosa")
librosa_onset_mock = types.ModuleType("librosa.onset")
librosa_beat_mock = types.ModuleType("librosa.beat")
librosa_onset_mock.onset_strength = MagicMock()
librosa_beat_mock.beat_track = MagicMock()
librosa_mock.onset = librosa_onset_mock
librosa_mock.beat = librosa_beat_mock
sys.modules["librosa"] = librosa_mock
sys.modules["librosa.onset"] = librosa_onset_mock
sys.modules["librosa.beat"] = librosa_beat_mock

# Clear any previously-cached src modules so they reimport with our mocks
for _key in list(sys.modules):
    if _key.startswith("src."):
        del sys.modules[_key]

# ---------------------------------------------------------------------------
# Now import the module under test
# ---------------------------------------------------------------------------
from src.utils import ensure_stereo, estimate_tempo, time_shift

# ===========================================================================
# ensure_stereo
# ===========================================================================


class TestEnsureStereo:
    def test_already_stereo_2d(self):
        audio = MockTensor(np.random.rand(2, 100))
        result = ensure_stereo(audio)
        assert result.shape == (2, 100)
        assert np.array_equal(result._data, audio._data)

    def test_already_stereo_3d(self):
        audio = MockTensor(np.random.rand(1, 2, 100))
        result = ensure_stereo(audio)
        assert result.shape == (1, 2, 100)
        assert np.array_equal(result._data, audio._data)

    def test_mono_2d_duplicated(self):
        audio = MockTensor(np.ones((1, 100)))
        result = ensure_stereo(audio)
        assert result.shape == (2, 100)
        assert np.array_equal(result._data[0], result._data[1])

    def test_mono_3d_duplicated(self):
        audio = MockTensor(np.ones((1, 1, 100)))
        result = ensure_stereo(audio)
        assert result.shape == (1, 2, 100)
        assert np.array_equal(result._data[0, 0], result._data[0, 1])

    def test_multichannel_2d_downmixed(self):
        data = np.array(
            [
                [1.0, 2.0, 3.0],
                [5.0, 6.0, 7.0],
                [9.0, 10.0, 11.0],
                [13.0, 14.0, 15.0],
            ]
        )
        audio = MockTensor(data)
        result = ensure_stereo(audio)
        assert result.shape == (2, 3)
        expected_mean = np.mean(data[:2], axis=0)
        np.testing.assert_allclose(result._data[0], expected_mean)
        np.testing.assert_allclose(result._data[1], expected_mean)

    def test_multichannel_3d_downmixed(self):
        data = np.random.rand(1, 4, 50).astype(np.float32)
        audio = MockTensor(data)
        result = ensure_stereo(audio)
        assert result.shape == (1, 2, 50)
        expected_mean = np.mean(data[:, :2, :], axis=1, keepdims=True)
        np.testing.assert_allclose(result._data[0, 0], expected_mean[0, 0])
        np.testing.assert_allclose(result._data[0, 1], expected_mean[0, 0])

    def test_invalid_1d_raises(self):
        audio = MockTensor(np.ones(100))
        with pytest.raises(ValueError, match="2 or 3 dimensions"):
            ensure_stereo(audio)

    def test_invalid_4d_raises(self):
        audio = MockTensor(np.ones((1, 1, 2, 100)))
        with pytest.raises(ValueError, match="2 or 3 dimensions"):
            ensure_stereo(audio)


# ===========================================================================
# estimate_tempo
# ===========================================================================


class TestEstimateTempo:
    def test_returns_tempo(self):
        librosa_onset_mock.onset_strength.return_value = np.ones(10)
        librosa_beat_mock.beat_track.return_value = (np.array([[120.0]]), None)

        waveform = MockTensor(np.random.rand(2, 22050))
        tempo = estimate_tempo(waveform, 22050)

        assert tempo == 120.0
        librosa_onset_mock.onset_strength.assert_called_once()
        librosa_beat_mock.beat_track.assert_called_once()

    def test_3d_input_gets_squeezed(self):
        librosa_onset_mock.onset_strength.reset_mock()
        librosa_beat_mock.beat_track.reset_mock()
        librosa_onset_mock.onset_strength.return_value = np.ones(10)
        librosa_beat_mock.beat_track.return_value = (np.array([[90.0]]), None)

        waveform = MockTensor(np.random.rand(1, 2, 22050))
        tempo = estimate_tempo(waveform, 22050)

        assert tempo == 90.0
        call_kwargs = librosa_onset_mock.onset_strength.call_args
        # After squeeze(0) the array should be 2D
        assert call_kwargs.kwargs["y"].ndim == 2

    def test_min_clamp(self):
        librosa_onset_mock.onset_strength.return_value = np.ones(10)
        librosa_beat_mock.beat_track.return_value = (np.array([[0.5]]), None)

        waveform = MockTensor(np.random.rand(2, 22050))
        tempo = estimate_tempo(waveform, 22050)

        assert tempo == 1.0

    def test_wrong_ndim_raises(self):
        waveform = MockTensor(np.random.rand(22050))
        with pytest.raises(TypeError, match="Expected waveform"):
            estimate_tempo(waveform, 22050)


# ===========================================================================
# time_shift
# ===========================================================================


class TestTimeShift:
    def _setup_stft_mocks(self, channels=2, freq=1025, time_frames=10, rate=1.0):
        """Configure stft/phase_vocoder/istft mocks for a call."""
        stft_result = MockTensor(np.zeros((channels, freq, time_frames)))
        # Ensure dtype matches torch.cfloat so the TypeError check passes
        stft_result.dtype = torch_mock.cfloat
        torch_mock.stft.reset_mock()
        torch_mock.istft.reset_mock()
        torchaudio_functional_mock.phase_vocoder.reset_mock()
        torch_mock.linspace.reset_mock()

        torch_mock.stft.return_value = stft_result

        stretched_time = math.ceil(time_frames / rate)
        stretched = MockTensor(np.zeros((channels, freq, stretched_time)))
        torchaudio_functional_mock.phase_vocoder.return_value = stretched

        output = MockTensor(np.zeros((channels, 44100)))
        torch_mock.istft.return_value = output

        return output

    def test_calls_phase_vocoder_with_rate(self):
        rate = 1.5
        self._setup_stft_mocks(rate=rate)
        waveform = MockTensor(np.zeros((2, 44100)))

        time_shift(waveform, rate)

        call_args = torchaudio_functional_mock.phase_vocoder.call_args
        assert call_args[0][1] == rate

    def test_default_hop_size(self):
        fft_size = 2048
        expected_hop = fft_size // 4
        self._setup_stft_mocks(rate=1.0)
        waveform = MockTensor(np.zeros((2, 44100)))

        time_shift(waveform, 1.0, fft_size=fft_size)

        stft_kwargs = torch_mock.stft.call_args
        assert stft_kwargs.kwargs["hop_length"] == expected_hop

    def test_custom_hop_size(self):
        self._setup_stft_mocks(rate=1.0)
        waveform = MockTensor(np.zeros((2, 44100)))

        time_shift(waveform, 1.0, hop_size=256)

        stft_kwargs = torch_mock.stft.call_args
        assert stft_kwargs.kwargs["hop_length"] == 256

    def test_non_complex_dtype_raises(self):
        stft_result = MockTensor(np.zeros((2, 1025, 10)))
        stft_result.dtype = "torch.float32"
        torch_mock.stft.return_value = stft_result

        waveform = MockTensor(np.zeros((2, 44100)))
        with pytest.raises(TypeError, match="Expected complex-valued STFT"):
            time_shift(waveform, 1.0)
