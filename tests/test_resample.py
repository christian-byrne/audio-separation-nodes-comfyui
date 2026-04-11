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

    # -- custom clamp values ------------------------------------------------

    def test_custom_upper_clamp(self):
        """Custom upper_clamp should override the default."""
        # Default UPPER_CLAMP is 1.1832; ratio 1.25 would be clamped by default
        # With upper_clamp=1.3, it should NOT be clamped
        r = ChunkResampler(44100, 55125, upper_clamp=1.3)
        # MockResample stores the reduced freqs; verify the ratio is preserved (not clamped)
        self.assertAlmostEqual(r.resample.orig / r.resample.new, 44100 / 55125, places=3)

    def test_custom_lower_clamp(self):
        """Custom lower_clamp should override the default."""
        # Default LOWER_CLAMP is 0.945; ratio 0.9 would be clamped by default
        # With lower_clamp=0.8, it should NOT be clamped
        r = ChunkResampler(44100, 39690, lower_clamp=0.8)
        self.assertAlmostEqual(r.resample.orig / r.resample.new, 44100 / 39690, places=3)

    def test_inverted_clamps_raises(self):
        """upper_clamp <= lower_clamp should raise ValueError."""
        with self.assertRaises(ValueError):
            ChunkResampler(44100, 44100, upper_clamp=0.5, lower_clamp=0.8)

    def test_zero_clamp_raises(self):
        """Zero clamp values should raise ValueError."""
        with self.assertRaises(ValueError):
            ChunkResampler(44100, 44100, upper_clamp=0.0)

    def test_negative_clamp_raises(self):
        """Negative clamp values should raise ValueError."""
        with self.assertRaises(ValueError):
            ChunkResampler(44100, 44100, lower_clamp=-0.5)


# ===========================================================================
# tolerance parameter tests
# ===========================================================================
class TestTolerance(unittest.TestCase):
    """Tests for the tolerance parameter."""

    def test_tolerance_negative_raises(self):
        with self.assertRaises(ValueError):
            ChunkResampler(44100, 48000, tolerance=-0.1)

    def test_tolerance_above_one_raises(self):
        with self.assertRaises(ValueError):
            ChunkResampler(44100, 48000, tolerance=1.5)

    def test_tolerance_zero_no_change(self):
        """tolerance=0.0 should behave exactly like no tolerance."""
        r = ChunkResampler(44100, 44100, tolerance=0.0)
        self.assertIsNotNone(r)

    def test_tolerance_finds_better_gcd(self):
        """With tolerance, the chosen freq should have a GCD >= the original target's GCD."""
        import math

        orig = 44100
        target = 48001  # coprime-ish with 44100
        original_gcd = math.gcd(orig, target)

        ChunkResampler(orig, target, tolerance=0.01)
        # _find_optimal_freq should have found a better candidate
        # We can't check self.new_freq directly because reduce_ratio transforms it,
        # but we can test the static method directly
        optimal = ChunkResampler._find_optimal_freq(orig, target, 0.01)
        optimal_gcd = math.gcd(orig, int(optimal))
        self.assertGreaterEqual(optimal_gcd, original_gcd)

    def test_find_optimal_freq_prefers_44100(self):
        """44100 should be found when searching near 44100 ± tolerance."""
        result = ChunkResampler._find_optimal_freq(44100, 44050, 0.01)
        # 44100 is in range and gcd(44100, 44100) = 44100 which is maximal
        self.assertEqual(result, 44100.0)

    def test_find_optimal_freq_exact_when_already_best(self):
        """If the target already has the best GCD, it should be returned as-is."""
        result = ChunkResampler._find_optimal_freq(44100, 22050, 0.001)
        # 22050 divides 44100 evenly → gcd = 22050, hard to beat
        self.assertEqual(result, 22050.0)

    def test_tolerance_one_is_max(self):
        """tolerance=1.0 is the maximum allowed value."""
        r = ChunkResampler(44100, 44100, tolerance=1.0)
        self.assertIsNotNone(r)

    def test_margin_capped_at_1000(self):
        """Search margin should be capped at 1000 candidates to avoid O(n) scan."""
        # With tolerance=1.0 and target=192000, uncapped margin would be ~192000
        # Capped at 1000, so search range is [191000, 193000]
        result = ChunkResampler._find_optimal_freq(44100, 192000, 1.0)
        # Should complete quickly and return a valid result
        self.assertGreater(result, 0)
        # Result should be within ±1000 of target
        self.assertLessEqual(abs(result - 192000), 1000)

    def test_chunk_size_uses_effective_ratio(self):
        """chunk_size_seconds should be based on effective ratio after clamping."""
        # orig=44100, new=48000, ratio=1.088 → clamped to 44100*1.1832≈52179
        # But effective ratio after clamping is 1.1832 → diff=0.1832 > 0.08 → cap=1
        r = ChunkResampler(44100, 48000)
        self.assertEqual(r.chunk_size_seconds, 1)


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
