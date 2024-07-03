import torch
import sys
import os
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from typing import Dict, Any


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

    FUNCTION = "run"
    RETURN_TYPES = ("AUDIO",)
    CATEGORY = "audio"

    def run(
        self,
        audio: dict,
        start_time: str = "0:00",
        end_time: str = "1:00",
    ) -> Dict[str, Any]:
        waveform: torch.Tensor = audio["waveform"]
        sample_rate: int = audio["sample_rate"]

        if ":" not in start_time:
            start_time = f"{start_time}:00"
        if ":" not in end_time:
            end_time = f"{end_time}:00"

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

        return (
            {
                "waveform": waveform[..., start_frame:end_frame],
                "sample_rate": sample_rate,
            },
        )
