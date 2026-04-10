from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# MockTensor – lightweight numpy-backed stand-in for torch.Tensor
# ---------------------------------------------------------------------------


class MockTensor:
    """Numpy-backed tensor mock supporting the operations used by separation.py."""

    def __init__(self, data=None, *, shape=None, device="cpu"):
        if data is not None:
            self._data = np.array(data, dtype=np.float32)
        elif shape is not None:
            self._data = np.zeros(shape, dtype=np.float32)
        else:
            self._data = np.zeros(0, dtype=np.float32)
        self.device = device

    # -- shape / indexing ---------------------------------------------------
    @property
    def shape(self):
        return self._data.shape

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, key):
        result = self._data[key]
        if isinstance(result, np.ndarray):
            t = MockTensor(result, device=self.device)
            return t
        return result

    def __setitem__(self, key, value):
        if isinstance(value, MockTensor):
            self._data[key] = value._data
        else:
            self._data[key] = value

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    # -- device movement ----------------------------------------------------
    def to(self, device):
        t = MockTensor(self._data.copy(), device=str(device))
        return t

    def cpu(self):
        return MockTensor(self._data.copy(), device="cpu")

    # -- shape manipulation -------------------------------------------------
    def squeeze(self, dim=None):
        return MockTensor(np.squeeze(self._data, axis=dim), device=self.device)

    def unsqueeze(self, dim):
        return MockTensor(np.expand_dims(self._data, axis=dim), device=self.device)

    # -- reductions ---------------------------------------------------------
    def mean(self, dim=None, keepdim=False):
        if dim is None:
            return MockTensor(np.array(self._data.mean(), dtype=np.float32), device=self.device)
        return MockTensor(self._data.mean(axis=dim, keepdims=keepdim), device=self.device)

    def std(self, *args, **kwargs):
        val = self._data.std()
        if val == 0:
            val = 1.0
        return MockTensor(np.array(val, dtype=np.float32), device=self.device)

    # -- arithmetic ---------------------------------------------------------
    def __sub__(self, other):
        o = other._data if isinstance(other, MockTensor) else other
        return MockTensor(self._data - o, device=self.device)

    def __truediv__(self, other):
        o = other._data if isinstance(other, MockTensor) else other
        return MockTensor(self._data / o, device=self.device)

    def __mul__(self, other):
        o = other._data if isinstance(other, MockTensor) else other
        return MockTensor(self._data * o, device=self.device)

    def __add__(self, other):
        o = other._data if isinstance(other, MockTensor) else other
        return MockTensor(self._data + o, device=self.device)

    def __iadd__(self, other):
        o = other._data if isinstance(other, MockTensor) else other
        self._data = self._data + o
        return self

    def __repr__(self):
        return f"MockTensor(shape={self.shape}, device={self.device!r})"

    def float(self):
        return MockTensor(self._data.astype(np.float32), device=self.device)

    @property
    def ndim(self):
        return self._data.ndim


# ---------------------------------------------------------------------------
# Mock torch / torchaudio / comfy – installed before importing separation.py
# ---------------------------------------------------------------------------


def _mock_zeros(*shape_args, device="cpu", **kwargs):
    if len(shape_args) == 1 and isinstance(shape_args[0], (tuple, list)):
        shape = tuple(shape_args[0])
    else:
        shape = tuple(shape_args)
    return MockTensor(shape=shape, device=device)


# --- torch -----------------------------------------------------------------
torch_mock = types.ModuleType("torch")
torch_mock.Tensor = MockTensor
torch_mock.zeros = _mock_zeros
torch_mock.device = str  # torch.device(x) → str(x)
torch_mock.no_grad = MagicMock(
    return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False))
)
torch_mock.nn = types.SimpleNamespace(Module=object)
torch_mock.cfloat = "cfloat"
sys.modules["torch"] = torch_mock

# --- torchaudio & submodules -----------------------------------------------
torchaudio_mock = types.ModuleType("torchaudio")
torchaudio_transforms = types.ModuleType("torchaudio.transforms")
torchaudio_pipelines = types.ModuleType("torchaudio.pipelines")


# Fade: record __init__ args and act as identity on __call__
class _MockFade:
    def __init__(self, fade_in_len=0, fade_out_len=0, fade_shape="linear"):
        self.fade_in_len = fade_in_len
        self.fade_out_len = fade_out_len
        self.fade_shape = fade_shape

    def __call__(self, x):
        return x  # identity – keeps tensor shapes unchanged


# Resample: record args, callable, supports .to(device)
class _MockResample:
    def __init__(self, orig_freq, new_freq):
        self.orig_freq = orig_freq
        self.new_freq = new_freq

    def to(self, device):
        return self

    def __call__(self, waveform):
        return waveform  # identity for simplicity


torchaudio_transforms.Fade = _MockFade
torchaudio_transforms.Resample = _MockResample

# Bundle mock – model w/ .sources and .forward
_MODEL_SR = 44100


class _MockModel:
    sources = ["bass", "drums", "other", "vocals"]

    def to(self, device):
        return self

    def forward(self, chunk):
        # chunk shape: [batch, channels, chunk_len]
        batch = chunk.shape[0]
        channels = chunk.shape[1]
        chunk_len = chunk.shape[2]
        return MockTensor(shape=(batch, len(self.sources), channels, chunk_len))


_mock_bundle = types.SimpleNamespace(
    get_model=lambda: _MockModel(),
    sample_rate=_MODEL_SR,
)
torchaudio_pipelines.HDEMUCS_HIGH_MUSDB_PLUS = _mock_bundle

torchaudio_mock.transforms = torchaudio_transforms
torchaudio_mock.pipelines = torchaudio_pipelines
sys.modules["torchaudio"] = torchaudio_mock
sys.modules["torchaudio.transforms"] = torchaudio_transforms
sys.modules["torchaudio.pipelines"] = torchaudio_pipelines

# --- comfy -----------------------------------------------------------------
comfy_mod = types.ModuleType("comfy")
comfy_mod.__path__ = []
comfy_mm = types.ModuleType("comfy.model_management")
comfy_mm.get_torch_device = lambda: "cuda"
comfy_mod.model_management = comfy_mm
sys.modules["comfy"] = comfy_mod
sys.modules["comfy.model_management"] = comfy_mm

# --- src._types (AUDIO is just a TypedDict; stub the import) ---------------
src_types = types.ModuleType("src._types")
src_types.AUDIO = dict
sys.modules["src._types"] = src_types

# --- src.utils – provide a real-enough ensure_stereo ----------------------
src_utils = types.ModuleType("src.utils")


def _ensure_stereo(audio):
    """If mono (1, N), duplicate to (2, N). Otherwise pass through."""
    if audio.ndim == 2 and audio.shape[0] == 1:
        return MockTensor(np.concatenate([audio._data, audio._data], axis=0))
    return audio


src_utils.ensure_stereo = _ensure_stereo
sys.modules["src.utils"] = src_utils

# Now import the module under test
separation = importlib.import_module("src.separation")
AudioSeparation = separation.AudioSeparation


# ===========================================================================
# Helpers
# ===========================================================================


def _make_audio(*, channels=2, frames=44100, sample_rate=44100):
    """Return an AUDIO dict with a MockTensor waveform [1, channels, frames]."""
    waveform = MockTensor(np.random.randn(1, channels, frames).astype(np.float32))
    return {"waveform": waveform, "sample_rate": sample_rate}


# ===========================================================================
# 1. INPUT_TYPES schema
# ===========================================================================


class TestInputTypes:
    def test_has_required_audio(self):
        schema = AudioSeparation.INPUT_TYPES()
        assert "required" in schema
        assert "audio" in schema["required"]
        assert schema["required"]["audio"] == ("AUDIO",)

    def test_has_optional_keys(self):
        schema = AudioSeparation.INPUT_TYPES()
        optional = schema["optional"]
        assert "chunk_fade_shape" in optional
        assert "chunk_length" in optional
        assert "chunk_overlap" in optional

    def test_fade_shape_options(self):
        schema = AudioSeparation.INPUT_TYPES()
        fade_choices = schema["optional"]["chunk_fade_shape"][0]
        assert set(fade_choices) == {"linear", "half_sine", "logarithmic", "exponential"}

    def test_chunk_length_is_float(self):
        schema = AudioSeparation.INPUT_TYPES()
        assert schema["optional"]["chunk_length"][0] == "FLOAT"

    def test_chunk_overlap_is_float(self):
        schema = AudioSeparation.INPUT_TYPES()
        assert schema["optional"]["chunk_overlap"][0] == "FLOAT"

    def test_chunk_length_default(self):
        schema = AudioSeparation.INPUT_TYPES()
        assert schema["optional"]["chunk_length"][1]["default"] == 10.0

    def test_chunk_overlap_default(self):
        schema = AudioSeparation.INPUT_TYPES()
        assert schema["optional"]["chunk_overlap"][1]["default"] == 0.1

    def test_fade_shape_default(self):
        schema = AudioSeparation.INPUT_TYPES()
        assert schema["optional"]["chunk_fade_shape"][1]["default"] == "linear"


# ===========================================================================
# 2. RETURN_TYPES / RETURN_NAMES
# ===========================================================================


class TestReturnMeta:
    def test_return_types(self):
        assert AudioSeparation.RETURN_TYPES == ("AUDIO", "AUDIO", "AUDIO", "AUDIO")

    def test_return_names(self):
        assert AudioSeparation.RETURN_NAMES == ("Bass", "Drums", "Other", "Vocals")

    def test_function_attr(self):
        assert AudioSeparation.FUNCTION == "main"

    def test_category(self):
        assert AudioSeparation.CATEGORY == "audio"


# ===========================================================================
# 3. sources_to_tuple
# ===========================================================================


class TestSourcesToTuple:
    def setup_method(self):
        self.node = AudioSeparation()
        self.node.model_sample_rate = _MODEL_SR

    def _make_sources(self, keys=None):
        keys = keys or ["bass", "drums", "other", "vocals"]
        return {k: MockTensor(shape=(2, 44100)) for k in keys}

    def test_valid_sources_returns_4_tuple(self):
        result = self.node.sources_to_tuple(self._make_sources())
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_output_order(self):
        sources = self._make_sources()
        result = self.node.sources_to_tuple(sources)
        for i, _name in enumerate(["bass", "drums", "other", "vocals"]):
            assert result[i]["sample_rate"] == _MODEL_SR

    def test_each_output_is_audio_dict(self):
        result = self.node.sources_to_tuple(self._make_sources())
        for audio in result:
            assert "waveform" in audio
            assert "sample_rate" in audio
            assert audio["sample_rate"] == _MODEL_SR

    def test_waveform_has_batch_dim(self):
        """sources_to_tuple calls .unsqueeze(0) → waveform should be 3-D."""
        result = self.node.sources_to_tuple(self._make_sources())
        for audio in result:
            assert audio["waveform"].ndim == 3
            assert audio["waveform"].shape[0] == 1

    def test_waveform_is_cpu(self):
        sources = {k: MockTensor(shape=(2, 44100), device="cuda") for k in ["bass", "drums", "other", "vocals"]}
        result = self.node.sources_to_tuple(sources)
        for audio in result:
            assert audio["waveform"].device == "cpu"

    def test_missing_source_raises(self):
        sources = self._make_sources(["bass", "drums", "other"])  # no vocals
        with pytest.raises(ValueError, match="Missing source vocals"):
            self.node.sources_to_tuple(sources)

    def test_missing_bass_raises(self):
        sources = self._make_sources(["drums", "other", "vocals"])
        with pytest.raises(ValueError, match="Missing source bass"):
            self.node.sources_to_tuple(sources)

    def test_extra_sources_ignored(self):
        sources = self._make_sources()
        sources["extra"] = MockTensor(shape=(2, 44100))
        result = self.node.sources_to_tuple(sources)
        assert len(result) == 4


# ===========================================================================
# 4. separate_sources – chunking logic
# ===========================================================================


class TestSeparateSources:
    def setup_method(self):
        self.node = AudioSeparation()
        self.model = _MockModel()

    def test_single_chunk_one_forward_call(self):
        """Audio shorter than one chunk → exactly one model.forward call."""
        sr = 44100
        frames = int(sr * 5)  # 5 seconds, segment=10 → fits in one chunk
        mix = MockTensor(shape=(1, 2, frames))

        self.model.forward = MagicMock(side_effect=lambda c: MockTensor(shape=(c.shape[0], 4, c.shape[1], c.shape[2])))

        result = self.node.separate_sources(self.model, mix, sr, segment=10.0, overlap=0.1)
        assert self.model.forward.call_count == 1
        assert result.shape == (1, 4, 2, frames)

    def test_multiple_chunks_multiple_forward_calls(self):
        """Audio much longer than chunk → multiple forward calls."""
        sr = 44100
        frames = int(sr * 30)  # 30 seconds with segment=10
        mix = MockTensor(shape=(1, 2, frames))

        call_count = 0

        def _forward(c):
            nonlocal call_count
            call_count += 1
            return MockTensor(shape=(c.shape[0], 4, c.shape[1], c.shape[2]))

        self.model.forward = _forward

        self.node.separate_sources(self.model, mix, sr, segment=10.0, overlap=0.1)
        assert call_count > 1

    def test_fade_parameters_set_correctly(self):
        """Fade should be created with fade_in_len=0 and the given fade_shape."""
        sr = 44100
        frames = int(sr * 5)
        mix = MockTensor(shape=(1, 2, frames))

        init_kwargs_list = []
        original_fade = _MockFade

        class _SpyFade(original_fade):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                # Capture the *initial* construction args before any mutation
                init_kwargs_list.append(dict(kwargs))

        with patch.object(separation, "Fade", _SpyFade):
            self.node.separate_sources(
                self.model,
                mix,
                sr,
                segment=10.0,
                overlap=0.1,
                chunk_fade_shape="half_sine",
            )

        assert len(init_kwargs_list) == 1
        init_kw = init_kwargs_list[0]
        assert init_kw["fade_in_len"] == 0
        assert init_kw["fade_shape"] == "half_sine"
        expected_overlap_frames = int(0.1 * sr)
        assert init_kw["fade_out_len"] == expected_overlap_frames

    def test_fade_in_set_after_first_chunk(self):
        """After the first chunk, fade_in_len should be set to overlap_frames."""
        sr = 44100
        frames = int(sr * 30)  # long enough for multiple chunks
        mix = MockTensor(shape=(1, 2, frames))

        created_fades = []
        original_fade = _MockFade

        class _SpyFade(original_fade):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                created_fades.append(self)

        with patch.object(separation, "Fade", _SpyFade):
            self.node.separate_sources(self.model, mix, sr, segment=10.0, overlap=0.1)

        fade = created_fades[0]
        expected_overlap = int(0.1 * sr)
        # After first iteration, fade_in_len should have been set
        assert fade.fade_in_len == expected_overlap

    def test_output_shape_matches_input(self):
        sr = 44100
        frames = int(sr * 8)
        mix = MockTensor(shape=(1, 2, frames))
        result = self.node.separate_sources(self.model, mix, sr, segment=10.0, overlap=0.1)
        assert result.shape == (1, 4, 2, frames)

    def test_device_none_uses_mix_device(self):
        """When device=None, should use mix.device."""
        sr = 44100
        frames = int(sr * 2)
        mix = MockTensor(shape=(1, 2, frames), device="cpu")
        result = self.node.separate_sources(self.model, mix, sr, device=None)
        assert result.device == "cpu"


# ===========================================================================
# 5. main flow – end-to-end with mocked pipeline
# ===========================================================================


class TestMainFlow:
    def setup_method(self):
        self.node = AudioSeparation()

    def test_ensure_stereo_called(self, monkeypatch):
        """ensure_stereo should be invoked on the waveform."""
        called = {}

        def spy_ensure_stereo(wav):
            called["yes"] = True
            return _ensure_stereo(wav)

        monkeypatch.setattr(separation, "ensure_stereo", spy_ensure_stereo)

        audio = _make_audio(channels=2, frames=44100, sample_rate=_MODEL_SR)
        self.node.main(audio)
        assert called.get("yes")

    def test_float_called_after_ensure_stereo(self, monkeypatch):
        """Fix #16/#22: waveform passed to the model should be float32, even if input is float64."""
        captured = {}

        original_sep = self.node.separate_sources

        def spy_separate(model, mix, sr, **kw):
            captured["dtype"] = mix._data.dtype
            return original_sep(model, mix, sr, **kw)

        monkeypatch.setattr(self.node, "separate_sources", spy_separate)

        # Use float64 input to verify it gets cast to float32
        data = np.random.randn(1, 2, 44100).astype(np.float64)
        audio = {"waveform": MockTensor(data), "sample_rate": _MODEL_SR}
        self.node.main(audio)
        assert captured["dtype"] == np.float32, "waveform should be float32 after .float() call"

    def test_resample_called_when_rates_differ(self, monkeypatch):
        """Resample should be instantiated when input SR != model SR."""
        created_resamplers = []
        original = _MockResample

        class _SpyResample(original):
            def __init__(self, orig, new):
                super().__init__(orig, new)
                created_resamplers.append((orig, new))

        monkeypatch.setattr(separation, "Resample", _SpyResample)

        audio = _make_audio(channels=2, frames=22050, sample_rate=22050)
        self.node.main(audio)
        assert len(created_resamplers) == 1
        assert created_resamplers[0] == (22050, _MODEL_SR)

    def test_resample_not_called_when_rates_match(self, monkeypatch):
        """Resample should NOT be called when sample rates match."""
        created_resamplers = []
        original = _MockResample

        class _SpyResample(original):
            def __init__(self, orig, new):
                super().__init__(orig, new)
                created_resamplers.append((orig, new))

        monkeypatch.setattr(separation, "Resample", _SpyResample)

        audio = _make_audio(channels=2, frames=44100, sample_rate=_MODEL_SR)
        self.node.main(audio)
        assert len(created_resamplers) == 0

    def test_normalization_applied(self, monkeypatch):
        """waveform should be normalized using ref = waveform.mean(0)."""
        captured = {}

        original_sep = self.node.separate_sources

        def spy_separate(model, mix, sr, **kw):
            # mix is waveform[None] after normalization — capture it
            captured["normalized_mix"] = mix
            return original_sep(model, mix, sr, **kw)

        monkeypatch.setattr(self.node, "separate_sources", spy_separate)

        data = np.full((1, 2, 44100), 5.0, dtype=np.float32)
        audio = {"waveform": MockTensor(data), "sample_rate": _MODEL_SR}
        self.node.main(audio)

        mix = captured["normalized_mix"]
        # After (waveform - ref.mean()) / ref.std(), the result should be
        # zero-centered. With constant input the std is ~0 so our MockTensor
        # std() returns 1.0 → result = (5-5)/1 = 0
        assert mix.shape[0] == 1  # batch dim from [None]

    def test_main_returns_4_tuple(self):
        audio = _make_audio(channels=2, frames=44100, sample_rate=_MODEL_SR)
        result = self.node.main(audio)
        assert isinstance(result, tuple)
        assert len(result) == 4
        for r in result:
            assert "waveform" in r
            assert "sample_rate" in r

    def test_main_output_sample_rate(self):
        audio = _make_audio(channels=2, frames=44100, sample_rate=_MODEL_SR)
        result = self.node.main(audio)
        for r in result:
            assert r["sample_rate"] == _MODEL_SR
