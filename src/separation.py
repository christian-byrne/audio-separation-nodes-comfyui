"""Credit: https://pytorch.org/audio/stable/tutorials/hybrid_demucs_tutorial.html"""

import torch
from torchaudio.transforms import Fade, Resample
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

import comfy.model_management

from typing import Dict, Tuple
from ._types import AUDIO
from .utils import ensure_stereo


class AudioSeparation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "chunk_fade_shape": (
                    [
                        "linear",
                        "half_sine",
                        "logarithmic",
                        "exponential",
                    ],
                    {
                        "default": "linear",
                        "tooltip": "Audio is split into segments (chunks) with overlapping areas to ensure smooth transitions. This setting controls the fade effect at these overlaps. Choose Linear for even fading, Half-Sine for a smooth curve, Logarithmic for a quick fade out and slow fade in, or Exponential for a slow fade out and quick fade in.",
                    },
                ),
                "chunk_length": (
                    "FLOAT",
                    {
                        "default": 10.0,
                        "tooltip": "The length of each segment (chunk) in seconds. Longer chunks may require more memory and MIGHT produce better results.",
                    },
                ),
                "chunk_overlap": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "tooltip": "The overlap between each segment (chunk) in seconds. A higher overlap may be necessary if chunks are too short or the audio changes rapidly.",
                    },
                ),
            },
        }

    FUNCTION = "main"
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("Bass", "Drums", "Other", "Vocals")
    CATEGORY = "audio"
    DESCRIPTION = "Separate audio into four sources: bass, drums, other, and vocals."

    def main(
        self,
        audio: AUDIO,
        chunk_fade_shape: str = "linear",
        chunk_length: float = 10.0,
        chunk_overlap: float = 0.1,
    ) -> Tuple[AUDIO, AUDIO, AUDIO, AUDIO]:

        device: torch.device = comfy.model_management.get_torch_device()
        waveform: torch.Tensor = audio["waveform"]
        waveform = waveform.squeeze(0).to(device)
        self.input_sample_rate_: int = audio["sample_rate"]

        bundle = HDEMUCS_HIGH_MUSDB_PLUS
        model: torch.nn.Module = bundle.get_model().to(device)
        self.model_sample_rate = bundle.sample_rate

        waveform = ensure_stereo(waveform)

        # Resample to model's expected sample rate
        if self.input_sample_rate_ != self.model_sample_rate:
            resample = Resample(self.input_sample_rate_, self.model_sample_rate).to(
                device
            )
            waveform = resample(waveform)

        ref = waveform.mean(0)
        waveform = (waveform - ref.mean()) / ref.std()  # Zs

        sources = self.separate_sources(
            model,
            waveform[None],
            self.model_sample_rate,
            segment=chunk_length,
            overlap=chunk_overlap,
            device=device,
            chunk_fade_shape=chunk_fade_shape,
        )[0]
        sources = sources * ref.std() + ref.mean()
        sources_list = model.sources
        sources = list(sources)

        return self.sources_to_tuple(dict(zip(sources_list, sources)))

    def sources_to_tuple(
        self, sources: Dict[str, torch.Tensor]
    ) -> Tuple[AUDIO, AUDIO, AUDIO, AUDIO]:

        output_order = ["bass", "drums", "other", "vocals"]
        outputs = []
        for source in output_order:
            if source not in sources:
                raise ValueError(f"Missing source {source} in the output")
            outputs.append(
                {
                    "waveform": sources[source].cpu().unsqueeze(0),
                    "sample_rate": self.model_sample_rate,
                }
            )
        return tuple(outputs)

    def separate_sources(
        self,
        model: torch.nn.Module,
        mix: torch.Tensor,
        sample_rate: int,
        segment: float = 10.0,
        overlap: float = 0.1,
        device: torch.device = None,
        chunk_fade_shape: str = "linear",
    ) -> torch.Tensor:
        """
        From: https://pytorch.org/audio/stable/tutorials/hybrid_demucs_tutorial.html

        Apply model to a given mixture. Use fade, and add segments together in order to add model segment by segment.

        Args:
            segment (int): segment length in seconds
            device (torch.device, str, or None): if provided, device on which to
                execute the computation, otherwise `mix.device` is assumed.
                When `device` is different from `mix.device`, only local computations will
                be on `device`, while the entire tracks will be stored on `mix.device`.
        """
        if device is None:
            device = mix.device
        else:
            device = torch.device(device)

        batch, channels, length = mix.shape

        chunk_len = int(sample_rate * segment * (1 + overlap))
        start = 0
        end = chunk_len
        overlap_frames = overlap * sample_rate
        fade = Fade(
            fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape=chunk_fade_shape
        )

        final = torch.zeros(batch, len(model.sources), channels, length, device=device)

        while start < length - overlap_frames:
            chunk = mix[:, :, start:end]
            with torch.no_grad():
                out = model.forward(chunk)
            out = fade(out)
            final[:, :, :, start:end] += out
            if start == 0:
                fade.fade_in_len = int(overlap_frames)
                start += int(chunk_len - overlap_frames)
            else:
                start += chunk_len
            end += chunk_len
            if end >= length:
                fade.fade_out_len = 0
        return final
