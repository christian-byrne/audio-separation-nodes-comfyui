import math
import numpy as np
import torch
import librosa

import torchaudio.functional as F

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

    def time_shift(
        self,
        waveform: torch.Tensor,
        rate: float,
        fft_size: int = 2048,
        hop_size: int = None,
        win_length: int = None,
    ) -> torch.Tensor:
        """
        Args:
            waveform (torch.Tensor): Time-domain input of shape [channels, frames]
            rate (float): rate to shift the waveform by
            fft_size (int): Size of the FFT to be used (power of 2)
            hop_size (int): Hop length for overlap (e.g., fft_size // 4)
            win_length (int): Window size (often equal to fft_size)

        Returns:
            torch.Tensor: Time-domain output of same shape/type as input [channels, frames]
            
        """
        if hop_size is None:
            hop_size = fft_size // 4
        if win_length is None:
            win_length = fft_size

        window = torch.hann_window(
            win_length, device=waveform.device
        )  # shape: [win_length]

        with torch.no_grad():
            complex_spectogram = torch.stft(
                waveform,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_length,
                window=window,
                return_complex=True,
            )  # shape: [channels, freq, time]

            if complex_spectogram.dtype != torch.cfloat:
                raise TypeError(f"Expected complex-valued STFT for phase vocoder, got dtype {complex_spectogram.dtype}")
            
            phase_advance = torch.linspace(0, math.pi * hop_size, complex_spectogram.shape[1])[
                ..., None
            ]  #  shape: [freq, 1]

            stretched_spectogram = F.phase_vocoder(
                complex_spectogram, rate, phase_advance
            )  # shape: [channels, freq, stretched_time]

            expected_time = math.ceil(complex_spectogram.shape[2] / rate)
            assert (
                abs(stretched_spectogram.shape[2] - expected_time) < 3
            ), f"Expected Time: {expected_time}, Stretched Time: {stretched_spectogram.shape[2]}"

            # Convert back to time basis with inverse STFT
            return torch.istft(
                stretched_spectogram,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_length,
                window=window,
            )  # shape: [channels, frames]

    def estimate_tempo(self, waveform: torch.Tensor, sample_rate: int) -> float:
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        if waveform.dim() != 2:
            raise TypeError(f"Expected waveform to be [channels, frames], got {waveform.shape}")

        onset_env = librosa.onset.onset_strength(
            y=waveform.numpy(),
            sr=sample_rate,
            aggregate=np.median,
        )

        tempo, _= librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sample_rate,
            tightness=110,
            sparse=False,
            trim=True,
        ) # [[channel 1 tempo], [channel 2 tempo], ...], _

        mean_tempo = np.mean(tempo.flatten())
        return max(mean_tempo, 1.0)

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

        rate_1 = avg_tempo / tempo_1
        rate_2 = avg_tempo / tempo_2

        waveform_1 = self.time_shift(waveform_1, rate_1)
        waveform_2 = self.time_shift(waveform_2, rate_2)

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
