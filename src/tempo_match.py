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
        """
        if hop_size is None:
            hop_size = fft_size // 4
        if win_length is None:
            win_length = fft_size

        with torch.no_grad():
            window = torch.hann_window(
                win_length, device=waveform.device
            )  # shape: [win_length]

            stft = torch.stft(
                waveform,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_length,
                window=window,
                return_complex=True,  # Need complex STFT for phase vocoder
            )  # shape: [channels, freq, time] (complex valued spectrogram) dtype: `torch.cfloat``

            assert (
                stft.dtype == torch.cfloat
            ), f"Expected complex STFT, got dtype {stft.dtype}"

            phase_advance = torch.linspace(0, math.pi * hop_size, stft.shape[1])[
                ..., None
            ]  #  shape: [freq, 1]

            stretched_stft = F.phase_vocoder(
                stft, rate, phase_advance
            )  # shape: [channels, freq, stretched_time]

            # New time should be (time in complex spectogram / rate)
            expected_time = math.ceil(stft.shape[2] / rate)
            assert (
                abs(stretched_stft.shape[2] - expected_time) < 3
            ), f"Expected Time: {expected_time}, Stretched Time: {stretched_stft.shape[2]}"

            # Use Inverse STFT to convert back to time domain
            return torch.istft(
                stretched_stft,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_length,
                window=window,
            )  # shape: [channels, frames]

    def estimate_tempo(self, waveform: torch.Tensor, sample_rate: int) -> float:
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        assert (
            waveform.dim() == 2
        ), f"Expected waveform to be [channels, frames], got {waveform.shape}"

        onset_env = librosa.onset.onset_strength(
            y=waveform.numpy(),
            sr=sample_rate,
            aggregate=np.median,  # Use median for tempo estimation, less sensitive to outliers
        )

        # Unused return value is the beat event locations array
        tempo, _ = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sample_rate,
            tightness=110,
            sparse=False,
            trim=True,
        )
        print(f"Estimated Tempo: {tempo}")

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
