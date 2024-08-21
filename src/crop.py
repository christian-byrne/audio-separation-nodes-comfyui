import torch

from typing import Tuple
from ._types import AUDIO


class AudioCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "start_time": (
                    "STRING",
                    {
                        "default": "0:00",
                    },
                ),
                "end_time": (
                    "STRING",
                    {
                        "default": "1:00",
                    },
                ),
            },
        }

    FUNCTION = "main"
    RETURN_TYPES = ("AUDIO",)
    CATEGORY = "audio"
    DESCRIPTION = "Crop (trim) audio to a specific start and end time."

    def main(
        self,
        audio: AUDIO,
        start_time: str = "0:00",
        end_time: str = "1:00",
    ) -> Tuple[AUDIO]:

        waveform: torch.Tensor = audio["waveform"]
        sample_rate: int = audio["sample_rate"]

        # Assume that no ":" in input means that the user is trying to specify seconds
        if ":" not in start_time:
            start_time = f"00:{start_time}"
        if ":" not in end_time:
            end_time = f"00:{end_time}"

        start_seconds_time = 60 * int(start_time.split(":")[0]) + int(
            start_time.split(":")[1]
        )
        start_frame = start_seconds_time * sample_rate
        if start_frame >= waveform.shape[-1]:
            start_frame = waveform.shape[-1] - 1

        end_seconds_time = 60 * int(end_time.split(":")[0]) + int(
            end_time.split(":")[1]
        )
        end_frame = end_seconds_time * sample_rate
        if end_frame >= waveform.shape[-1]:
            end_frame = waveform.shape[-1] - 1
        if start_frame < 0:
            start_frame = 0
        if end_frame < 0:
            end_frame = 0

        if start_frame > end_frame:
            raise ValueError(
                "AudioCrop: Start time must be less than end time and be within the audio length."
            )

        return (
            {
                "waveform": waveform[..., start_frame:end_frame],
                "sample_rate": sample_rate,
            },
        )
