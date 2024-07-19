import torch
from torchaudio.transforms import Resample

import comfy.model_management

from typing import Union, Tuple


class ChunkResampler:
    """
    a larger lowpass_filter_width results in a larger resampling kernel, and therefore increases computation time for both the kernel computation and convolution

    using sinc_interp_kaiser results in longer computation times than the default sinc_interp_hann because it is more complex to compute the intermediate window values

    a large GCD between the sample and resample rate will result in a simplification that allows for a smaller kernel and faster kernel computation.

    """

    def __init__(
        self,
        orig_freq: Union[int, float],
        new_freq: Union[int, float],
        chunk_size_seconds: int = 2,
    ):
        if orig_freq < 0 or new_freq < 0:
            raise ValueError("Frequencies must be positive.")

        self.UPPER_CLAMP = 1.1832
        self.LOWER_CLAMP = 0.945
        self.orig_freq = orig_freq
        self.new_freq = new_freq

        self.chunk_size_seconds = int(chunk_size_seconds)
        change_ratio = new_freq / orig_freq
        if change_ratio > self.UPPER_CLAMP:
            self.new_freq = self.orig_freq * self.UPPER_CLAMP
        elif change_ratio < self.LOWER_CLAMP:
            self.new_freq = self.orig_freq * self.LOWER_CLAMP

        diff = abs(1 - change_ratio)
        if diff > 0.08:
            self.chunk_size_seconds = min(self.chunk_size_seconds, 1)
        elif diff > 0.002:
            self.chunk_size_seconds = min(self.chunk_size_seconds, 2)
        else:
            self.chunk_size_seconds = min(self.chunk_size_seconds, 4)

        # If the frequencies are float, try to convert to int while
        # maintaining ratio (https://github.com/pytorch/audio/issues/1487).
        self.orig_freq, self.new_freq = ChunkResampler.reduce_ratio(orig_freq, new_freq)
        self.device = comfy.model_management.get_torch_device()
        self.resample = Resample(self.orig_freq, self.new_freq).to(self.device)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.to(self.device)

        with torch.no_grad():
            chunks = torch.split(
                waveform, int(self.orig_freq * self.chunk_size_seconds), dim=-1
            )
            resampled_chunks = [self.resample(chunk) for chunk in chunks]
            resampled_waveform = torch.cat(resampled_chunks, dim=-1)

        return resampled_waveform.to("cpu")

    @staticmethod
    def reduce_ratio(
        num1: Union[float, int], num2: Union[float, int]
    ) -> Tuple[int, int]:
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
