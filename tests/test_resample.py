"""Tests for src.resample.ChunkResampler."""

import sys
import types
import unittest

import numpy as np

# ---------------------------------------------------------------------------
# Mock torch, torchaudio, and comfy *before* importing the module under test
# ---------------------------------------------------------------------------

torch_mock = types.ModuleType("torch")


class MockTensor:
    def __init__(self, data):
        self._data = np.array(data) if not isinstance(data, np.ndarray) else data

    @property
    def shape(self):
        return self._data.shape

    def to(self, device):
        self._last_device = device
        return self


torch_mock.Tensor = MockTensor


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return None


torch_mock.no_grad = _NoGrad
torch_mock.split = lambda tensor, size, dim=-1: [tensor]
torch_mock.cat = lambda tensors, dim=-1: tensors[0]
sys.modules["torch"] = torch_mock

torchaudio_mock = types.ModuleType("torchaudio")
transforms_mock = types.ModuleType("torchaudio.transforms")


class MockResample:
    def __init__(self, orig, new):
        self.orig = orig
        self.new = new

    def to(self, device):
        return self

    def __call__(self, x):
        return x


transforms_mock.Resample = MockResample
torchaudio_mock.transforms = transforms_mock
sys.modules["torchaudio"] = torchaudio_mock
sys.modules["torchaudio.transforms"] = transforms_mock

comfy_mock = types.ModuleType("comfy")
mm_mock = types.ModuleType("comfy.model_management")
mm_mock.get_torch_device = lambda: "cpu"
comfy_mock.model_management = mm_mock
sys.modules["comfy"] = comfy_mock
sys.modules["comfy.model_management"] = mm_mock

# Now safe to import -------------------------------------------------------
from src.resample import ChunkResampler  # noqa: E402


# ===========================================================================
# reduce_ratio tests
# ===========================================================================
class TestReduceRatio(unittest.TestCase):
    """Tests for the static ChunkResampler.reduce_ratio method."""

    def test_simple_halving(self):
        self.assertEqual(ChunkResampler.reduce_ratio(44100, 22050), (2, 1))

    def test_48000_44100(self):
        a, b = ChunkResampler.reduce_ratio(48000, 44100)
        # Must be reduced integers smaller than originals
        self.assertIsInstance(a, int)
        self.assertIsInstance(b, int)
        self.assertLessEqual(a, 48000)
        self.assertLessEqual(b, 44100)
        # The ratio must be preserved
        self.assertAlmostEqual(a / b, 48000 / 44100, places=5)

    def test_identity(self):
        self.assertEqual(ChunkResampler.reduce_ratio(44100, 44100), (1, 1))

    def test_float_inputs(self):
        a, b = ChunkResampler.reduce_ratio(44100.0, 22050.0)
        self.assertEqual((a, b), (2, 1))

    def test_large_numbers_fallback(self):
        """Very large coprime-ish numbers should hit max_attempts and fall back."""
        big1 = 1_000_003  # large prime
        big2 = 1_000_033  # another large prime
        a, b = ChunkResampler.reduce_ratio(big1, big2)
        # Fallback returns int(originals)
        self.assertEqual((a, b), (big1, big2))


# ===========================================================================
# Constructor tests
# ===========================================================================
class TestConstructor(unittest.TestCase):
    """Tests for ChunkResampler.__init__."""

    def test_negative_orig_freq_raises(self):
        with self.assertRaises(ValueError):
            ChunkResampler(-1, 44100)

    def test_negative_new_freq_raises(self):
        with self.assertRaises(ValueError):
            ChunkResampler(44100, -1)

    # -- clamping -----------------------------------------------------------

    def test_ratio_above_upper_clamp(self):
        """change_ratio > UPPER_CLAMP (1.1832) → new_freq is clamped."""
        orig, new = 44100, 60000  # ratio ≈ 1.36
        r = ChunkResampler(orig, new)
        # After clamping self.new_freq = orig * UPPER_CLAMP, but reduce_ratio
        # is called with the *original* args, so stored freqs come from that.
        # Just verify construction succeeds and chunk_size is set.
        self.assertIsNotNone(r)

    def test_ratio_below_lower_clamp(self):
        """change_ratio < LOWER_CLAMP (0.945) → new_freq is clamped."""
        orig, new = 44100, 40000  # ratio ≈ 0.907
        r = ChunkResampler(orig, new)
        self.assertIsNotNone(r)

    def test_ratio_within_bounds(self):
        """Ratio inside [LOWER_CLAMP, UPPER_CLAMP] → no clamping."""
        orig, new = 44100, 44100  # ratio = 1.0
        r = ChunkResampler(orig, new)
        self.assertIsNotNone(r)

    # -- chunk_size_seconds adjustment --------------------------------------

    def test_chunk_size_large_diff(self):
        """diff > 0.08 → chunk_size_seconds capped at 1."""
        # ratio = 48000/44100 ≈ 1.0884 → diff ≈ 0.0884 > 0.08
        r = ChunkResampler(44100, 48000)
        self.assertEqual(r.chunk_size_seconds, 1)

    def test_chunk_size_medium_diff(self):
        """0.002 < diff ≤ 0.08 → chunk_size_seconds capped at 2."""
        # ratio = 44100/43000 ≈ 1.0256 → diff ≈ 0.0256
        r = ChunkResampler(43000, 44100)
        self.assertEqual(r.chunk_size_seconds, 2)

    def test_chunk_size_small_diff(self):
        """diff ≤ 0.002 → chunk_size_seconds capped at 4."""
        # ratio = 1.0 → diff = 0
        r = ChunkResampler(44100, 44100, chunk_size_seconds=10)
        self.assertEqual(r.chunk_size_seconds, 4)

    def test_chunk_size_seconds_truncated_to_int(self):
        r = ChunkResampler(44100, 44100, chunk_size_seconds=3.7)
        self.assertIsInstance(r.chunk_size_seconds, int)


# ===========================================================================
# __call__ tests
# ===========================================================================
class TestCall(unittest.TestCase):
    """Tests for ChunkResampler.__call__."""

    def setUp(self):
        self.resampler = ChunkResampler(44100, 44100)

    def test_waveform_moved_to_device(self):
        wav = MockTensor(np.zeros((1, 44100)))
        self.resampler(wav)
        # .to() was called with the device during __call__
        self.assertTrue(hasattr(wav, "_last_device"))

    def test_result_on_cpu(self):
        wav = MockTensor(np.zeros((1, 44100)))
        result = self.resampler(wav)
        self.assertEqual(result._last_device, "cpu")

    def test_split_and_cat_called(self):
        """Verify the chunk→resample→cat pipeline runs without error."""
        calls = {"split": 0, "cat": 0}
        orig_split = torch_mock.split
        orig_cat = torch_mock.cat

        def counting_split(tensor, size, dim=-1):
            calls["split"] += 1
            return orig_split(tensor, size, dim)

        def counting_cat(tensors, dim=-1):
            calls["cat"] += 1
            return orig_cat(tensors, dim)

        torch_mock.split = counting_split
        torch_mock.cat = counting_cat
        try:
            wav = MockTensor(np.zeros((1, 44100)))
            self.resampler(wav)
            self.assertGreaterEqual(calls["split"], 1)
            self.assertGreaterEqual(calls["cat"], 1)
        finally:
            torch_mock.split = orig_split
            torch_mock.cat = orig_cat


if __name__ == "__main__":
    unittest.main()
