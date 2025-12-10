import torch
from typing import List, Tuple, Any

from ._types import AUDIO
from .utils import ensure_stereo


class AudioStitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_list": ("AUDIO_LIST",),
            },
            "optional": {
                "force_stereo": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If True, convert all clips to stereo before stitching to avoid channel mismatches.",
                    },
                ),
            },
        }

    FUNCTION = "main"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("Audio",)
    CATEGORY = "audio"
    DESCRIPTION = "Concatenate a list of AUDIO clips into a single longer clip."

    def main(self, audio_list: List[AUDIO], force_stereo: bool = True) -> Tuple[AUDIO]:
        if not isinstance(audio_list, list) or len(audio_list) == 0:
            raise ValueError("AudioStitch: 'audio_list' must be a non-empty list.")

        # Validate sample rates and dimensions
        sample_rate = audio_list[0]["sample_rate"]
        first_wave = audio_list[0]["waveform"]
        if not isinstance(first_wave, torch.Tensor):
            raise TypeError("AudioStitch: waveform must be a torch.Tensor.")
        ndims = first_wave.ndim

        processed: List[torch.Tensor] = []
        for idx, a in enumerate(audio_list):
            if a["sample_rate"] != sample_rate:
                raise ValueError(
                    f"AudioStitch: sample_rate mismatch at index {idx}: {a['sample_rate']} != {sample_rate}"
                )
            w = a["waveform"]
            if w.ndim != ndims:
                raise ValueError(
                    f"AudioStitch: waveform dim mismatch at index {idx}: {w.ndim} != {ndims}"
                )
            if force_stereo:
                w = ensure_stereo(w)
            processed.append(w)

        # Confirm channel count alignment
        channels = [t.shape[-2] if t.ndim >= 2 else 1 for t in processed]
        if len(set(channels)) != 1:
            raise ValueError(
                f"AudioStitch: channel count mismatch after processing: {channels}. Disable 'force_stereo' only if all match."
            )

        # Concatenate along time dimension (last axis)
        stitched = torch.cat(processed, dim=-1)

        return ({"waveform": stitched, "sample_rate": sample_rate},)

