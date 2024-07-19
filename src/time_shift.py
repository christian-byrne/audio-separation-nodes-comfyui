from .utils import time_shift

from typing import Tuple
from ._types import AUDIO


class TimeShift:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "rate": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0}),
            },
        }

    FUNCTION = "main"
    RETURN_TYPES = ("AUDIO",)
    CATEGORY = "audio"

    def main(
        self,
        audio: AUDIO,
        rate: float,
    ) -> Tuple[AUDIO, AUDIO]:
        waveform = audio["waveform"].squeeze(0)
        sample_rate = audio["sample_rate"]
        rate = min(max(rate, 0.1), 10.0)
        shifted = time_shift(waveform, rate)

        return (
            {
                "waveform": shifted.unsqueeze(0),
                "sample_rate": sample_rate,
            },
        )
