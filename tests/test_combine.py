from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# MockTensor – numpy-backed tensor that supports .shape, .to(), slicing, math
# ---------------------------------------------------------------------------


class MockTensor:
    def __init__(self, data: np.ndarray, device: str = "cpu"):
        self._data = np.asarray(data, dtype=np.float64)
        self.device = device

    @property
    def shape(self):
        return self._data.shape

    def to(self, device):
        return MockTensor(self._data, device=str(device))

    def __getitem__(self, key):
        return MockTensor(self._data[key], device=self.device)

    def __add__(self, other):
        return MockTensor(self._data + _unwrap(other), device=self.device)

    def __sub__(self, other):
        return MockTensor(self._data - _unwrap(other), device=self.device)

    def __mul__(self, other):
        return MockTensor(self._data * _unwrap(other), device=self.device)

    def __truediv__(self, other):
        return MockTensor(self._data / _unwrap(other), device=self.device)

    def __eq__(self, other):
        return np.array_equal(self._data, _unwrap(other))

    def numpy(self):
        return self._data


def _unwrap(obj):
    if isinstance(obj, MockTensor):
        return obj._data
    return obj


# ---------------------------------------------------------------------------
# Module-level mocks — must be registered before importing src.combine
# ---------------------------------------------------------------------------

# torch
torch_mock = types.ModuleType("torch")
torch_mock.Tensor = MockTensor
sys.modules["torch"] = torch_mock

# torchaudio + torchaudio.transforms
torchaudio_mock = types.ModuleType("torchaudio")
torchaudio_mock.__path__ = []
transforms_mock = types.ModuleType("torchaudio.transforms")

_resample_calls: list[dict] = []


class _MockResample:
    def __init__(self, orig_freq: int, new_freq: int):
        self.orig_freq = orig_freq
        self.new_freq = new_freq

    def to(self, device):
        return self

    def __call__(self, waveform: MockTensor) -> MockTensor:
        _resample_calls.append({"orig_freq": self.orig_freq, "new_freq": self.new_freq})
        # Return the waveform unchanged (resampling logic is not under test)
        return waveform


transforms_mock.Resample = _MockResample
torchaudio_mock.transforms = transforms_mock
sys.modules["torchaudio"] = torchaudio_mock
sys.modules["torchaudio.transforms"] = transforms_mock

# comfy + comfy.model_management
comfy_mock = types.ModuleType("comfy")
comfy_mock.__path__ = []
model_management_mock = types.ModuleType("comfy.model_management")
model_management_mock.get_torch_device = lambda: "cpu"
comfy_mock.model_management = model_management_mock
sys.modules["comfy"] = comfy_mock
sys.modules["comfy.model_management"] = model_management_mock

# Clear any previously-cached src modules so they reimport with our mocks
for _key in list(sys.modules):
    if _key.startswith("src."):
        del sys.modules[_key]

# Now import the module under test
module = importlib.import_module("src.combine")
AudioCombine = module.AudioCombine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _audio(values, sample_rate: int = 44100) -> dict:
    """Create an AUDIO dict with a 1×1×N MockTensor waveform."""
    arr = np.array(values, dtype=np.float64).reshape(1, 1, -1)
    return {"waveform": MockTensor(arr), "sample_rate": sample_rate}


@pytest.fixture(autouse=True)
def _clear_resample_calls():
    _resample_calls.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInputTypesSchema:
    def test_required_keys(self):
        schema = AudioCombine.INPUT_TYPES()
        assert "audio_1" in schema["required"]
        assert "audio_2" in schema["required"]

    def test_method_options(self):
        schema = AudioCombine.INPUT_TYPES()
        method_opts = schema["optional"]["method"]
        assert set(method_opts[0]) == {"add", "mean", "subtract", "multiply", "divide"}
        assert method_opts[1]["default"] == "add"


class TestClassAttributes:
    def test_return_types(self):
        assert AudioCombine.RETURN_TYPES == ("AUDIO",)

    def test_category(self):
        assert AudioCombine.CATEGORY == "audio"

    def test_function_name(self):
        assert AudioCombine.FUNCTION == "main"


class TestSameSampleRateAdd:
    def test_basic_add(self):
        a1 = _audio([1.0, 2.0, 3.0])
        a2 = _audio([4.0, 5.0, 6.0])
        (result,) = AudioCombine().main(a1, a2, method="add")
        np.testing.assert_array_equal(result["waveform"].numpy().flatten(), [5.0, 7.0, 9.0])
        assert result["sample_rate"] == 44100


class TestAllMethods:
    @pytest.mark.parametrize(
        "method, expected",
        [
            ("add", [5.0, 7.0, 9.0]),
            ("subtract", [-3.0, -3.0, -3.0]),
            ("multiply", [4.0, 10.0, 18.0]),
            ("divide", [0.25, 0.4, 0.5]),
            ("mean", [2.5, 3.5, 4.5]),
        ],
    )
    def test_method(self, method, expected):
        a1 = _audio([1.0, 2.0, 3.0])
        a2 = _audio([4.0, 5.0, 6.0])
        (result,) = AudioCombine().main(a1, a2, method=method)
        np.testing.assert_allclose(result["waveform"].numpy().flatten(), expected)


class TestDifferentLengths:
    def test_truncation_to_shorter(self):
        a1 = _audio([1.0, 2.0, 3.0, 4.0, 5.0])
        a2 = _audio([10.0, 20.0, 30.0])
        (result,) = AudioCombine().main(a1, a2, method="add")
        out = result["waveform"].numpy().flatten()
        assert len(out) == 3
        np.testing.assert_array_equal(out, [11.0, 22.0, 33.0])

    def test_truncation_second_longer(self):
        a1 = _audio([1.0, 2.0])
        a2 = _audio([10.0, 20.0, 30.0, 40.0])
        (result,) = AudioCombine().main(a1, a2, method="add")
        out = result["waveform"].numpy().flatten()
        assert len(out) == 2
        np.testing.assert_array_equal(out, [11.0, 22.0])


class TestDifferentSampleRates:
    def test_lower_rate_gets_resampled_upward(self):
        a1 = _audio([1.0, 2.0, 3.0], sample_rate=22050)
        a2 = _audio([4.0, 5.0, 6.0], sample_rate=44100)
        (result,) = AudioCombine().main(a1, a2, method="add")
        assert result["sample_rate"] == 44100
        assert len(_resample_calls) == 1
        assert _resample_calls[0] == {"orig_freq": 22050, "new_freq": 44100}

    def test_higher_rate_stays_when_second_is_lower(self):
        a1 = _audio([1.0, 2.0, 3.0], sample_rate=48000)
        a2 = _audio([4.0, 5.0, 6.0], sample_rate=16000)
        (result,) = AudioCombine().main(a1, a2, method="add")
        assert result["sample_rate"] == 48000
        assert len(_resample_calls) == 1
        assert _resample_calls[0] == {"orig_freq": 16000, "new_freq": 48000}

    def test_resampled_waveform_moved_to_cpu_first_lower(self):
        """Fix #23: resampled waveform_1 should be moved back to cpu."""
        a1 = _audio([1.0, 2.0, 3.0], sample_rate=22050)
        a1["waveform"] = a1["waveform"].to("cuda")
        a2 = _audio([4.0, 5.0, 6.0], sample_rate=44100)
        a2["waveform"] = a2["waveform"].to("cuda")
        (result,) = AudioCombine().main(a1, a2, method="add")
        assert result["waveform"].device == "cpu"

    def test_resampled_waveform_moved_to_cpu_second_lower(self):
        """Fix #23: resampled waveform_2 should be moved back to cpu."""
        a1 = _audio([1.0, 2.0, 3.0], sample_rate=48000)
        a1["waveform"] = a1["waveform"].to("cuda")
        a2 = _audio([4.0, 5.0, 6.0], sample_rate=16000)
        a2["waveform"] = a2["waveform"].to("cuda")
        (result,) = AudioCombine().main(a1, a2, method="add")
        assert result["waveform"].device == "cpu"


class TestUnsupportedMethod:
    def test_raises_value_error(self):
        a1 = _audio([1.0])
        a2 = _audio([1.0])
        with pytest.raises(ValueError, match="Unsupported combine method"):
            AudioCombine().main(a1, a2, method="max")
