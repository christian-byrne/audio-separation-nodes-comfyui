from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 1. sys.path setup (kept from original)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 2. MockTensor – lightweight stand-in when torch is unavailable
# ---------------------------------------------------------------------------
class MockTensor:
    """Minimal tensor mock that supports shape manipulation, arithmetic, and
    conversion methods used by ComfyUI audio nodes."""

    def __init__(
        self,
        shape: tuple[int, ...] = (1, 1, 44100),
        dtype: str = "float32",
        device: str = "cpu",
    ):
        self._shape = tuple(shape)
        self.dtype = dtype
        self.device = device

    # -- shape / dim --------------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    def dim(self) -> int:
        return len(self._shape)

    # -- device / dtype conversions -----------------------------------------
    def to(self, *_args, **_kwargs) -> MockTensor:
        return self

    def float(self) -> MockTensor:
        return MockTensor(self._shape, dtype="float32", device=self.device)

    def cpu(self) -> MockTensor:
        return MockTensor(self._shape, dtype=self.dtype, device="cpu")

    # -- shape manipulation -------------------------------------------------
    def squeeze(self, dim: int = 0) -> MockTensor:
        s = list(self._shape)
        if 0 <= dim < len(s) and s[dim] == 1 or dim < 0 and (dim + len(s)) >= 0 and s[dim] == 1:
            s.pop(dim)
        return MockTensor(tuple(s), dtype=self.dtype, device=self.device)

    def unsqueeze(self, dim: int) -> MockTensor:
        s = list(self._shape)
        if dim < 0:
            dim = len(s) + 1 + dim
        s.insert(dim, 1)
        return MockTensor(tuple(s), dtype=self.dtype, device=self.device)

    def mean(self, dim: int | None = None, keepdim: bool = False) -> MockTensor:
        if dim is None:
            return MockTensor((1,), dtype=self.dtype, device=self.device)
        s = list(self._shape)
        if keepdim:
            s[dim] = 1
        else:
            s.pop(dim)
        return MockTensor(tuple(s), dtype=self.dtype, device=self.device)

    # -- numpy conversion ---------------------------------------------------
    def numpy(self) -> np.ndarray:
        return np.zeros(self._shape, dtype=np.float32)

    # -- arithmetic ---------------------------------------------------------
    def _binop(self, other: object) -> MockTensor:
        return MockTensor(self._shape, dtype=self.dtype, device=self.device)

    __add__ = __radd__ = _binop
    __sub__ = __rsub__ = _binop
    __mul__ = __rmul__ = _binop
    __truediv__ = __rtruediv__ = _binop

    # -- slicing (e.g. tensor[..., start:end]) ------------------------------
    def __getitem__(self, key: object) -> MockTensor:
        if not isinstance(key, tuple):
            key = (key,)

        new_shape = list(self._shape)
        real_idx = 0
        for k in key:
            if k is Ellipsis:
                real_idx = len(new_shape) - (len(key) - 1 - list(key).index(Ellipsis))
                continue
            if isinstance(k, slice):
                start = k.start or 0
                stop = k.stop if k.stop is not None else new_shape[real_idx]
                new_shape[real_idx] = max(stop - start, 0)
            real_idx += 1

        return MockTensor(tuple(new_shape), dtype=self.dtype, device=self.device)

    def __repr__(self) -> str:
        return f"MockTensor(shape={self._shape}, dtype={self.dtype})"


# ---------------------------------------------------------------------------
# 3. Autouse session fixture – stub comfy ecosystem modules only when absent
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True, scope="session")
def mock_comfy_modules(tmp_path_factory: pytest.TempPathFactory):
    """Injects lightweight stubs for comfy-ecosystem modules that are not
    available in the test environment.  Skips any module that is *already*
    present in ``sys.modules`` so that test files which set up their own
    mocks (e.g. ``test_audio_video_combine_node.py``) keep working."""

    _tmp = tmp_path_factory.mktemp("comfy_temp")
    installed: list[str] = []

    # NOTE: torch, torchaudio, and librosa are NOT mocked here.
    # Each test file sets up its own mocks with the fidelity it needs.

    # -- comfy + comfy.model_management -------------------------------------
    if "comfy" not in sys.modules:
        comfy_stub = types.ModuleType("comfy")
        comfy_stub.__path__ = []
        sys.modules["comfy"] = comfy_stub
        installed.append("comfy")

    if "comfy.model_management" not in sys.modules:
        mm_stub = types.ModuleType("comfy.model_management")
        mm_stub.get_torch_device = lambda: "cpu"
        sys.modules["comfy.model_management"] = mm_stub
        if hasattr(sys.modules.get("comfy", None), "__path__"):
            sys.modules["comfy"].model_management = mm_stub
        installed.append("comfy.model_management")

    # -- comfy_api (and sub-modules) ----------------------------------------
    if "comfy_api" not in sys.modules:
        comfy_api_stub = types.ModuleType("comfy_api")
        comfy_api_stub.__path__ = []
        sys.modules["comfy_api"] = comfy_api_stub
        installed.append("comfy_api")

    # -- folder_paths -------------------------------------------------------
    if "folder_paths" not in sys.modules:
        fp_stub = types.ModuleType("folder_paths")
        fp_stub.get_temp_directory = lambda: str(_tmp)
        sys.modules["folder_paths"] = fp_stub
        installed.append("folder_paths")

    yield

    # Teardown: remove only what *we* installed
    for mod_name in installed:
        sys.modules.pop(mod_name, None)


# ---------------------------------------------------------------------------
# 4. make_audio fixture factory
# ---------------------------------------------------------------------------
def _make_tensor(shape, dtype="float32"):
    """Return a real torch.Tensor if torch is usable, else a MockTensor."""
    try:
        import torch as _torch

        if hasattr(_torch, "zeros") and callable(_torch.zeros):
            # Guard against our own stub leaking through
            t = _torch.zeros(*shape)
            if isinstance(t, MockTensor):
                raise TypeError
            return t
    except Exception:
        pass
    return MockTensor(shape, dtype=dtype)


@pytest.fixture()
def make_audio():
    """Factory fixture that returns AUDIO TypedDicts.

    Usage::

        def test_something(make_audio):
            audio = make_audio(shape=(1, 2, 44100), sample_rate=44100)
    """

    def _factory(
        shape: tuple[int, ...] = (1, 1, 44100),
        sample_rate: int = 44100,
        dtype: str = "float32",
    ) -> dict:
        return {
            "waveform": _make_tensor(shape, dtype=dtype),
            "sample_rate": sample_rate,
        }

    return _factory
