import torch

from typing import Tuple, List
from ._types import AUDIO


class AudioSplitEqual:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "segments": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": "Number of equal segments to split the audio into (1-8).",
                    },
                )
            },
        }

    FUNCTION = "main"
    # Fixed maximum outputs; unused outputs will be None
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = (
        "Segment 1",
        "Segment 2",
        "Segment 3",
        "Segment 4",
        "Segment 5",
        "Segment 6",
        "Segment 7",
        "Segment 8",
    )
    CATEGORY = "audio"
    DESCRIPTION = "Split an audio clip into N equal segments and output up to 8 segments."

    def main(self, audio: AUDIO, segments: int = 2) -> Tuple[AUDIO, ...]:
        waveform: torch.Tensor = audio["waveform"]
        sample_rate: int = audio["sample_rate"]

        # Normalize and clamp segments
        try:
            n = int(segments)
        except Exception:
            n = 2
        n = max(1, min(8, n))

        total_frames = waveform.shape[-1]
        base = total_frames // n
        rem = total_frames % n

        # Distribute remainder: first `rem` segments get one extra frame
        sizes: List[int] = [(base + 1 if i < rem else base) for i in range(n)]

        # Create cumulative indices for slicing
        indices = [0]
        for s in sizes:
            indices.append(indices[-1] + s)

        outputs: List[AUDIO] = []
        for i in range(n):
            start, end = indices[i], indices[i + 1]
            seg_waveform = waveform[..., start:end]
            outputs.append({"waveform": seg_waveform, "sample_rate": sample_rate})

        # Pad remaining outputs with None to satisfy fixed RETURN_TYPES
        while len(outputs) < 8:
            outputs.append(None)

        return tuple(outputs)


class AudioSplitEqualList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "segments": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Number of equal segments to split the audio into (1-64).",
                    },
                )
            },
        }

    FUNCTION = "main"
    RETURN_TYPES = ("AUDIO_LIST",)
    RETURN_NAMES = ("Segments",)
    CATEGORY = "audio"
    DESCRIPTION = "Split an audio clip into N equal segments and return a list of AUDIO objects."

    def main(self, audio: AUDIO, segments: int = 2):
        waveform: torch.Tensor = audio["waveform"]
        sample_rate: int = audio["sample_rate"]

        try:
            n = int(segments)
        except Exception:
            n = 2
        n = max(1, min(64, n))

        total_frames = waveform.shape[-1]
        base = total_frames // n
        rem = total_frames % n

        sizes = [(base + 1 if i < rem else base) for i in range(n)]

        indices = [0]
        for s in sizes:
            indices.append(indices[-1] + s)

        outputs = []
        for i in range(n):
            start, end = indices[i], indices[i + 1]
            seg_waveform = waveform[..., start:end]
            outputs.append({"waveform": seg_waveform, "sample_rate": sample_rate})

        return (outputs,)
