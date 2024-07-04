"""TODO: Can use TimeStretch instead"""

import torch
import librosa

from .resample import ChunkResampler

from typing import Tuple
from ._types import AUDIO


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
        self.UPPER_CLAMP = 1.2
        self.LOWER_CLAMP = 0.955

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
        audio_1: AUDIO,
        audio_2: AUDIO,
    ) -> Tuple[AUDIO, AUDIO]:
        waveform_1: torch.Tensor = audio_1["waveform"].squeeze(0)
        input_sample_rate_1: int = audio_1["sample_rate"]

        waveform_2: torch.Tensor = audio_2["waveform"].squeeze(0)
        input_sample_rate_2: int = audio_2["sample_rate"]

        tempo_1 = self.estimate_tempo(waveform_1, input_sample_rate_1)
        tempo_2 = self.estimate_tempo(waveform_2, input_sample_rate_2)
        avg_tempo = (tempo_1 + tempo_2) / 2

        new_freq_1 = avg_tempo * input_sample_rate_1 / tempo_1
        new_freq_2 = avg_tempo * input_sample_rate_2 / tempo_2

        if new_freq_1 != input_sample_rate_1:
            waveform_1 = ChunkResampler(input_sample_rate_1, new_freq_1)(waveform_1)
        if new_freq_2 != input_sample_rate_2:
            waveform_2 = ChunkResampler(input_sample_rate_2, new_freq_2)(waveform_2)

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
