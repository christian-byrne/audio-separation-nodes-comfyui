from typing import TypedDict
from torch import Tensor


class AUDIO(TypedDict):
    """
    A dictionary containing audio data and sample rate.

    Required Fields:
        waveform (torch.Tensor): A tensor containing the audio data. The shape should be [Batch, Channels, frames].
        sample_rate (int): The sample rate of the audio data.
    """

    waveform: Tensor
    sample_rate: int
