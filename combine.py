import torch
from torchaudio.transforms import Resample

import comfy.model_management

from typing import Tuple
from _types import AUDIO


class AudioCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_1": ("AUDIO",),
                "audio_2": ("AUDIO",),
            },
            "optional": {
                "method": (
                    ["add", "mean", "subtract", "multiply", "divide"],
                    {"default": "add"},
                )
            },
        }

    FUNCTION = "main"
    RETURN_TYPES = ("AUDIO",)
    CATEGORY = "audio"

    def main(
        self,
        audio_1: AUDIO,
        audio_2: AUDIO,
        method: str = "add",
    ) -> Tuple[AUDIO]:

        waveform_1: torch.Tensor = audio_1["waveform"]
        input_sample_rate_1: int = audio_1["sample_rate"]

        waveform_2: torch.Tensor = audio_2["waveform"]
        input_sample_rate_2: int = audio_2["sample_rate"]

        # Resample the audio if the sample rates are different
        if input_sample_rate_1 != input_sample_rate_2:
            device = comfy.model_management.get_torch_device()
            if input_sample_rate_1 < input_sample_rate_2:
                resample = Resample(input_sample_rate_1, input_sample_rate_2).to(
                    device
                )
                waveform_1 = resample(waveform_1.to(device))
                output_sample_rate = input_sample_rate_2
            else:
                resample = Resample(input_sample_rate_2, input_sample_rate_1).to(
                    device
                )
                waveform_2 = resample(waveform_2.to(device))
                output_sample_rate = input_sample_rate_1
        else:
            output_sample_rate = input_sample_rate_1

        # Ensure the audio is the same length
        min_length = min(waveform_1.shape[-1], waveform_2.shape[-1])
        waveform_1 = waveform_1[..., :min_length]
        waveform_2 = waveform_2[..., :min_length]

        # Combine the audio
        if method == "add":
            waveform = waveform_1 + waveform_2
        elif method == "subtract":
            waveform = waveform_1 - waveform_2
        elif method == "multiply":
            waveform = waveform_1 * waveform_2
        elif method == "divide":
            waveform = waveform_1 / waveform_2
        elif method == "mean":
            waveform = (waveform_1 + waveform_2) / 2

        return (
            {
                "waveform": waveform.to("cpu"),
                "sample_rate": output_sample_rate,
            },
        )
