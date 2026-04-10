"""Tests for node registration in __init__.py."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock all heavy dependencies before importing any src modules.
#
# Use sys.modules[...] = ... (not setdefault) so these mocks take precedence
# even if conftest.py or another test file already inserted a partial mock.
# ---------------------------------------------------------------------------

# -- torch (needs nn.Module, device, Tensor, and several top-level functions) --
_torch = types.ModuleType("torch")
_torch.Tensor = object
_torch.device = str  # just needs to be a type
_torch.stft = lambda *a, **kw: None
_torch.istft = lambda *a, **kw: None
_torch.hann_window = lambda *a, **kw: None
_torch.cat = lambda *a, **kw: None
_torch.zeros = lambda *a, **kw: None
_torch.linspace = lambda *a, **kw: None
_torch.cfloat = "torch.cfloat"
_torch.no_grad = type("_NoGrad", (), {"__enter__": lambda s: s, "__exit__": lambda s, *a: None})

_nn = types.ModuleType("torch.nn")
_nn.Module = type(
    "Module",
    (),
    {"to": lambda self, d: self, "forward": lambda self, x: x, "sources": ["bass", "drums", "other", "vocals"]},  # noqa: E501
)
_torch.nn = _nn

sys.modules["torch"] = _torch
sys.modules["torch.nn"] = _nn

# -- torchaudio (transforms, pipelines, functional) --
_torchaudio = types.ModuleType("torchaudio")
_torchaudio.save = lambda *a, **kw: None

_transforms = types.ModuleType("torchaudio.transforms")
_transforms.Fade = object
_transforms.Resample = object
_torchaudio.transforms = _transforms

_pipelines = types.ModuleType("torchaudio.pipelines")
_pipelines.HDEMUCS_HIGH_MUSDB_PLUS = types.SimpleNamespace(get_model=lambda: None, sample_rate=44100)
_torchaudio.pipelines = _pipelines

_functional = types.ModuleType("torchaudio.functional")
_functional.phase_vocoder = lambda *a, **kw: None
_torchaudio.functional = _functional

sys.modules["torchaudio"] = _torchaudio
sys.modules["torchaudio.transforms"] = _transforms
sys.modules["torchaudio.pipelines"] = _pipelines
sys.modules["torchaudio.functional"] = _functional

# -- comfy --
_comfy = types.ModuleType("comfy")
_comfy.__path__ = []
_model_mgmt = types.ModuleType("comfy.model_management")
_model_mgmt.get_torch_device = lambda: "cpu"
_comfy.model_management = _model_mgmt
sys.modules["comfy"] = _comfy
sys.modules["comfy.model_management"] = _model_mgmt

# -- comfy_api --
_comfy_api = types.ModuleType("comfy_api")
_comfy_api.__path__ = []
sys.modules["comfy_api"] = _comfy_api

_input_impl = types.ModuleType("comfy_api.input_impl")
_input_impl.VideoFromFile = object
sys.modules["comfy_api.input_impl"] = _input_impl
_comfy_api.input_impl = _input_impl

_input_mod = types.ModuleType("comfy_api.input")
_input_mod.__path__ = []
_video_types = types.ModuleType("comfy_api.input.video_types")
_video_types.VideoInput = object
sys.modules["comfy_api.input"] = _input_mod
sys.modules["comfy_api.input.video_types"] = _video_types
_input_mod.video_types = _video_types
_comfy_api.input = _input_mod

# -- moviepy --
_moviepy = types.ModuleType("moviepy")
_moviepy_editor = types.ModuleType("moviepy.editor")
_moviepy_editor.VideoFileClip = object
_moviepy_editor.AudioFileClip = object
_moviepy.editor = _moviepy_editor
_moviepy.VideoFileClip = object
_moviepy.AudioFileClip = object
sys.modules.setdefault("moviepy", _moviepy)
sys.modules.setdefault("moviepy.editor", _moviepy_editor)

# -- folder_paths --
_folder_paths = types.ModuleType("folder_paths")
_folder_paths.get_temp_directory = lambda: "/tmp"
sys.modules.setdefault("folder_paths", _folder_paths)

# -- librosa --
_librosa = types.ModuleType("librosa")
_librosa_onset = types.ModuleType("librosa.onset")
_librosa_onset.onset_strength = lambda *a, **kw: np.zeros(10)
_librosa.onset = _librosa_onset
_librosa_beat = types.ModuleType("librosa.beat")
_librosa_beat.beat_track = lambda *a, **kw: (np.array([[120.0]]), None)
_librosa.beat = _librosa_beat
sys.modules["librosa"] = _librosa
sys.modules["librosa.onset"] = _librosa_onset
sys.modules["librosa.beat"] = _librosa_beat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_NODE_KEYS = [
    "AudioSeparation",
    "AudioCrop",
    "AudioCombine",
    "AudioTempoMatch",
    "AudioVideoCombine",
    "AudioSpeedShift",
    "AudioGetTempo",
]

# NODE_CLASS_MAPPINGS key -> fallback source module used by __init__.py
_KEY_TO_SRC = {
    "AudioSeparation": "src.separation",
    "AudioCrop": "src.crop",
    "AudioCombine": "src.combine",
    "AudioTempoMatch": "src.tempo_match",
    "AudioVideoCombine": "src.combine_video_with_audio",
    "AudioSpeedShift": "src.time_shift",
    "AudioGetTempo": "src.get_tempo",
}


def _reload_init():
    """Force-reload __init__.py so NODE_CLASS_MAPPINGS is rebuilt."""
    # Re-install our mocks — other test files may have overwritten them
    sys.modules["torch"] = _torch
    sys.modules["torch.nn"] = _nn
    sys.modules["torchaudio"] = _torchaudio
    sys.modules["torchaudio.transforms"] = _transforms
    sys.modules["torchaudio.pipelines"] = _pipelines
    sys.modules["torchaudio.functional"] = _functional
    sys.modules["librosa"] = _librosa
    sys.modules["librosa.onset"] = _librosa_onset
    sys.modules["librosa.beat"] = _librosa_beat

    # Clear cached src submodules so Python re-executes them on import
    for mod_name in list(sys.modules):
        if mod_name.startswith("src.") or mod_name == "src":
            sys.modules.pop(mod_name, None)

    # Remove the top-level __init__ itself so reload actually re-runs it
    sys.modules.pop("__init__", None)

    import __init__ as pkg  # noqa: E0611

    return pkg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAllNodesRegister:
    """All nodes register when imports succeed."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.pkg = _reload_init()

    def test_all_keys_present(self):
        for key in ALL_NODE_KEYS:
            assert key in self.pkg.NODE_CLASS_MAPPINGS, f"{key} missing from NODE_CLASS_MAPPINGS"

    def test_no_extra_keys(self):
        assert set(self.pkg.NODE_CLASS_MAPPINGS.keys()) == set(ALL_NODE_KEYS)

    def test_values_are_not_none(self):
        for key in ALL_NODE_KEYS:
            assert self.pkg.NODE_CLASS_MAPPINGS[key] is not None


class TestGracefulDegradation:
    """When a specific import fails, that node is missing but others still register."""

    @pytest.mark.parametrize("broken_key", ALL_NODE_KEYS)
    def test_single_broken_import(self, broken_key):
        src_module = _KEY_TO_SRC[broken_key]

        # Re-install our mocks — other test files may have overwritten them
        sys.modules["torch"] = _torch
        sys.modules["torch.nn"] = _nn
        sys.modules["torchaudio"] = _torchaudio
        sys.modules["torchaudio.transforms"] = _transforms
        sys.modules["torchaudio.pipelines"] = _pipelines
        sys.modules["torchaudio.functional"] = _functional
        sys.modules["librosa"] = _librosa
        sys.modules["librosa.onset"] = _librosa_onset
        sys.modules["librosa.beat"] = _librosa_beat

        # Clear all src modules so nothing is cached
        for mod_name in list(sys.modules):
            if mod_name.startswith("src.") or mod_name == "src":
                sys.modules.pop(mod_name, None)
        sys.modules.pop("__init__", None)

        # Poison the target module: __getattr__ raises ImportError so
        # ``from src.<mod> import <Class>`` fails.
        class _Raiser(types.ModuleType):
            def __getattr__(self, name):
                raise ImportError(f"mocked failure for {src_module}")

        sys.modules[src_module] = _Raiser(src_module)

        try:
            import __init__ as pkg  # noqa: E0611

            assert broken_key not in pkg.NODE_CLASS_MAPPINGS, (
                f"{broken_key} should NOT be registered when its import fails"
            )

            for key in ALL_NODE_KEYS:
                if key == broken_key:
                    continue
                assert key in pkg.NODE_CLASS_MAPPINGS, f"{key} should still register when only {broken_key} is broken"
        finally:
            # Clean up so other tests get a fresh slate
            sys.modules.pop(src_module, None)
            sys.modules.pop("__init__", None)


class TestNodeClassAttributes:
    """Each registered class has the attributes ComfyUI expects."""

    REQUIRED_ATTRS = ["INPUT_TYPES", "FUNCTION", "RETURN_TYPES", "CATEGORY"]

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.pkg = _reload_init()

    @pytest.mark.parametrize("key", ALL_NODE_KEYS)
    def test_has_required_attributes(self, key):
        cls = self.pkg.NODE_CLASS_MAPPINGS[key]
        for attr in self.REQUIRED_ATTRS:
            assert hasattr(cls, attr), f"{key} missing attribute {attr}"

    @pytest.mark.parametrize("key", ALL_NODE_KEYS)
    def test_input_types_is_callable(self, key):
        cls = self.pkg.NODE_CLASS_MAPPINGS[key]
        assert callable(cls.INPUT_TYPES)

    @pytest.mark.parametrize("key", ALL_NODE_KEYS)
    def test_function_is_string(self, key):
        cls = self.pkg.NODE_CLASS_MAPPINGS[key]
        assert isinstance(cls.FUNCTION, str)

    @pytest.mark.parametrize("key", ALL_NODE_KEYS)
    def test_return_types_is_tuple(self, key):
        cls = self.pkg.NODE_CLASS_MAPPINGS[key]
        assert isinstance(cls.RETURN_TYPES, tuple)

    @pytest.mark.parametrize("key", ALL_NODE_KEYS)
    def test_category_is_string(self, key):
        cls = self.pkg.NODE_CLASS_MAPPINGS[key]
        assert isinstance(cls.CATEGORY, str)
