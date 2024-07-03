"""TODO: Can use TimeStretch instead"""

import os
import sys
import torch
import torchaudio
import torchaudio.transforms
import librosa

from typing import Any, Dict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import comfy.model_management


class TempoMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_1": ("AUDIO",),
                "audio_2": ("AUDIO",),
            },
        }

    FUNCTION = "main"
    RETURN_TYPES = ("AUDIO", "AUDIO")
    CATEGORY = "audio"

    def __init__(self):
        # TODO
        self.UPPER_CLAMP = 1.05
        self.LOWER_CLAMP = 0.75

    def estimate_tempo(self, waveform: torch.Tensor, sample_rate: int) -> float:
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        assert waveform.dim() == 2
        waveform = waveform.numpy()
        onset_env = librosa.onset.onset_strength(y=waveform, sr=sample_rate)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sample_rate)
        return max(tempo[0][0], 1.0)

    def main(
        self,
        audio_1: dict,
        audio_2: dict,
    ) -> Dict[str, Any]:
        waveform_1: torch.Tensor = audio_1["waveform"].squeeze(0)
        input_sample_rate_1: int = audio_1["sample_rate"]

        waveform_2: torch.Tensor = audio_2["waveform"].squeeze(0)
        input_sample_rate_2: int = audio_2["sample_rate"]

        device = comfy.model_management.get_torch_device()
        tempo_1 = self.estimate_tempo(waveform_1, input_sample_rate_1)
        tempo_2 = self.estimate_tempo(waveform_2, input_sample_rate_2)
        print(f"Tempo 1: {tempo_1}, Tempo 2: {tempo_2}")
        avg_tempo = (tempo_1 + tempo_2) / 2
        print(f"Avg Tempo: {avg_tempo}")

        new_freq_1 = avg_tempo * input_sample_rate_1 / tempo_1
        if new_freq_1 < 1:
            new_freq_1 = 1
        new_freq_2 = avg_tempo * input_sample_rate_2 / tempo_2
        if new_freq_2 < 1:
            new_freq_2 = 1

        print(f"New Freq 1: {new_freq_1}, New Freq 2: {new_freq_2}")
        new_freq_1 = min(
            avg_tempo * input_sample_rate_1 / tempo_1,
            input_sample_rate_1 * self.UPPER_CLAMP,
        )
        # Clamp for now until
        new_freq_1 = max(new_freq_1, input_sample_rate_1 * self.LOWER_CLAMP)
        new_freq_2 = min(
            avg_tempo * input_sample_rate_2 / tempo_2,
            input_sample_rate_2 * self.UPPER_CLAMP,
        )
        new_freq_2 = max(new_freq_2, input_sample_rate_2 * self.LOWER_CLAMP)
        print(f"Clamped New Freq 1: {new_freq_1}, Clamped New Freq 2: {new_freq_2}")

        if new_freq_1 != input_sample_rate_1:
            # TODO
            waveform_1 = waveform_1.to(device)
            resampler1 = torchaudio.transforms.Resample(
                orig_freq=input_sample_rate_1, new_freq=int(new_freq_1)
            ).to(device)
            waveform_1 = resampler1(waveform_1).to("cpu")
        if new_freq_2 != input_sample_rate_2:
            waveform_2 = waveform_2.to(device)
            resampler2 = torchaudio.transforms.Resample(
                orig_freq=input_sample_rate_2, new_freq=int(new_freq_2)
            ).to(device)
            waveform_2 = resampler2(waveform_2).to("cpu")

        return (
            {
                "waveform": waveform_1.unsqueeze(0),
                "sample_rate": input_sample_rate_1,
            },
            {
                "waveform": waveform_2.unsqueeze(0),
                "sample_rate": input_sample_rate_2,
            },
        )
