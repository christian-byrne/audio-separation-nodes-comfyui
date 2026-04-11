from __future__ import annotations

import math

import comfy.model_management
import torch
from torchaudio.transforms import Resample


class ChunkResampler:
    """
    a larger lowpass_filter_width results in a larger resampling kernel, and therefore increases
    computation time for both the kernel computation and convolution

    using sinc_interp_kaiser results in longer computation times than the default sinc_interp_hann
    because it is more complex to compute the intermediate window values

    a large GCD between the sample and resample rate will result in a simplification that allows
    for a smaller kernel and faster kernel computation.

    """

    DEFAULT_UPPER_CLAMP = 1.1832
    DEFAULT_LOWER_CLAMP = 0.945

    def __init__(
        self,
        orig_freq: int | float,
        new_freq: int | float,
        chunk_size_seconds: int = 2,
        tolerance: float = 0.0,
        upper_clamp: float | None = None,
        lower_clamp: float | None = None,
    ):
        if orig_freq < 0 or new_freq < 0:
            raise ValueError("Frequencies must be positive.")
        if tolerance < 0.0 or tolerance > 1.0:
            raise ValueError("Tolerance must be between 0.0 and 1.0.")

        self.upper_clamp = upper_clamp if upper_clamp is not None else self.DEFAULT_UPPER_CLAMP
        self.lower_clamp = lower_clamp if lower_clamp is not None else self.DEFAULT_LOWER_CLAMP
        if self.upper_clamp <= 0 or self.lower_clamp <= 0:
            raise ValueError("Clamp values must be positive.")
        if self.upper_clamp <= self.lower_clamp:
            raise ValueError("upper_clamp must be greater than lower_clamp.")

        self.orig_freq = orig_freq
        self.new_freq = new_freq

        self.chunk_size_seconds = int(chunk_size_seconds)
        change_ratio = new_freq / orig_freq
        if change_ratio > self.upper_clamp:
            self.new_freq = self.orig_freq * self.upper_clamp
        elif change_ratio < self.lower_clamp:
            self.new_freq = self.orig_freq * self.lower_clamp

        if tolerance > 0.0:
            self.new_freq = self._find_optimal_freq(round(self.orig_freq), self.new_freq, tolerance)

        effective_ratio = self.new_freq / self.orig_freq
        diff = abs(1 - effective_ratio)
        if diff > 0.08:
            self.chunk_size_seconds = min(self.chunk_size_seconds, 1)
        elif diff > 0.002:
            self.chunk_size_seconds = min(self.chunk_size_seconds, 2)
        else:
            self.chunk_size_seconds = min(self.chunk_size_seconds, 4)

        # If the frequencies are float, try to convert to int while
        # maintaining ratio (https://github.com/pytorch/audio/issues/1487).
        self.orig_freq, self.new_freq = ChunkResampler.reduce_ratio(self.orig_freq, self.new_freq)
        self.device = comfy.model_management.get_torch_device()
        self.resample = Resample(self.orig_freq, self.new_freq).to(self.device)

    @staticmethod
    def _find_optimal_freq(orig_freq: int, target_freq: float, tolerance: float) -> float:
        """Find a frequency near *target_freq* that shares a large GCD with *orig_freq*.

        Searches integer candidates within ``target_freq * (1 ± tolerance)`` and
        returns the one whose GCD with *orig_freq* is largest, producing a
        smaller resampling kernel.
        """
        target = int(round(target_freq))
        margin = min(max(1, int(target * tolerance)), 1000)
        lo = max(1, target - margin)
        hi = target + margin

        best_freq = target
        best_gcd = math.gcd(orig_freq, target)

        for candidate in range(lo, hi + 1):
            g = math.gcd(orig_freq, candidate)
            if g > best_gcd:
                best_gcd = g
                best_freq = candidate

        return float(best_freq)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.to(self.device)

        with torch.no_grad():
            chunks = torch.split(waveform, int(self.orig_freq * self.chunk_size_seconds), dim=-1)
            resampled_chunks = [self.resample(chunk) for chunk in chunks]
            resampled_waveform = torch.cat(resampled_chunks, dim=-1)

        return resampled_waveform.to("cpu")

    @staticmethod
    def reduce_ratio(num1: float | int, num2: float | int) -> tuple[int, int]:
        """Reduces a ratio to its smallest **integer** form.

        Args:
            num1 (int): The numerator.
            num2 (int): The denominator.

        Returns:
            Tuple[int, int]: The reduced ratio.
        """
        originals = (num1, num2)
        num1 = round(num1, 1)  # increase for more precision
        num2 = round(num2, 1)

        if isinstance(num1, float) or isinstance(num2, float):
            while (isinstance(num1, float) and not num1.is_integer()) or (
                isinstance(num2, float) and not num2.is_integer()
            ):
                num1 *= 10
                num2 *= 10

        scaled_originals = (num1, num2)
        num1, num2 = int(num1), int(num2)

        if num1 < num2:
            num1, num2 = num2, num1

        attempts = 0
        max_attempts = 128 if max(num1, num2) < 200_000 else 32
        while num2 != 0:
            if attempts == max_attempts:
                return int(originals[0]), int(originals[1])
            num1, num2 = num2, num1 % num2
            attempts += 1

        gcd = num1

        new_num1 = int(scaled_originals[0] / gcd)
        new_num2 = int(scaled_originals[1] / gcd)

        if new_num1 > originals[0] or new_num2 > originals[1]:
            return int(originals[0]), int(originals[1])

        return new_num1, new_num2
