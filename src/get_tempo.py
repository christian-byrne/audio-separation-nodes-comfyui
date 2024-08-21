from .utils import estimate_tempo

from typing import Tuple
from ._types import AUDIO


class GetTempo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
        }

    FUNCTION = "main"
    RETURN_TYPES = ("STRING", "FLOAT", "INTEGER")
    RETURN_NAMES = ("tempo_string", "tempo_float", "tempo_integer")
    CATEGORY = "audio"
    DESCRIPTION = "Get the tempo (BPM) of audio using onset detection."

    def main(
        self,
        audio: AUDIO,
    ) -> Tuple[AUDIO, AUDIO]:
        waveform = audio["waveform"].squeeze(0)
        sample_rate = audio["sample_rate"]
        tempo = estimate_tempo(waveform, sample_rate)

        return (f"{int(round(tempo))}", tempo, int(tempo))
