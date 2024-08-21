from .utils import estimate_tempo, time_shift

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
    DESCRIPTION = "Match the tempo of two audio tracks by time-stretching them both to match the average tempo between them. E.g., if one audio track is 120 BPM and the other is 100 BPM, both will be time-stretched to 110 BPM."

    def main(
        self,
        audio_1: AUDIO,
        audio_2: AUDIO,
    ) -> Tuple[AUDIO, AUDIO]:
        waveform_1 = audio_1["waveform"].squeeze(0)
        input_sample_rate_1 = audio_1["sample_rate"]

        waveform_2 = audio_2["waveform"].squeeze(0)
        input_sample_rate_2 = audio_2["sample_rate"]

        tempo_1 = estimate_tempo(waveform_1, input_sample_rate_1)
        tempo_2 = estimate_tempo(waveform_2, input_sample_rate_2)
        avg_tempo = (tempo_1 + tempo_2) / 2

        rate_1 = avg_tempo / tempo_1
        rate_2 = avg_tempo / tempo_2

        waveform_1 = time_shift(waveform_1, rate_1)
        waveform_2 = time_shift(waveform_2, rate_2)

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
