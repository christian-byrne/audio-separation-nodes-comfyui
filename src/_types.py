from typing import TypedDict
from torch import Tensor


class AUDIO(TypedDict):
    """
    Required Fields:
        waveform (torch.Tensor): A tensor containing the audio data. Shape: [Batch, Channels, frames].
        sample_rate (int): The sample rate of the audio data.
    """

    waveform: Tensor
    sample_rate: int
