import librosa
import torch
import math
import torchaudio.functional as F
import numpy as np


def time_shift(
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
            raise TypeError(
                f"Expected complex-valued STFT for phase vocoder, got dtype {complex_spectogram.dtype}"
            )

        phase_advance = torch.linspace(
            0, math.pi * hop_size, complex_spectogram.shape[1]
        )[
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


def estimate_tempo(waveform: torch.Tensor, sample_rate: int) -> float:
    if waveform.dim() == 3:
        waveform = waveform.squeeze(0)
    if waveform.dim() != 2:
        raise TypeError(
            f"Expected waveform to be [channels, frames], got {waveform.shape}"
        )

    onset_env = librosa.onset.onset_strength(
        y=waveform.numpy(),
        sr=sample_rate,
        aggregate=np.median,
    )

    tempo, _ = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sample_rate,
        tightness=110,
        sparse=False,
        trim=True,
    )  # [[channel 1 tempo], [channel 2 tempo], ...], _

    mean_tempo = np.mean(tempo.flatten())
    return max(mean_tempo, 1.0)
