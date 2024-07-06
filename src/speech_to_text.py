
"""
>>> import torchaudio
>>> from torchaudio.pipelines import CONVTASNET_BASE_LIBRI2MIX
>>> import torch
>>>
>>> # Build the separation model.
>>> model = CONVTASNET_BASE_LIBRI2MIX.get_model()
>>> 100%|███████████████████████████████|19.1M/19.1M [00:04<00:00, 4.93MB/s]
>>>
>>> # Instantiate the test set of Libri2Mix dataset.
>>> dataset = torchaudio.datasets.LibriMix("/home/datasets/", subset="test")
>>>
>>> # Apply source separation on mixture audio.
>>> for i, data in enumerate(dataset):
>>>     sample_rate, mixture, clean_sources = data
>>>     # Make sure the shape of input suits the model requirement.
>>>     mixture = mixture.reshape(1, 1, -1)
>>>     estimated_sources = model(mixture)
>>>     score = si_snr_pit(estimated_sources, clean_sources) # for demonstration
>>>     print(f"Si-SNR score is : {score}.)
>>>     break
>>> Si-SNR score is : 16.24.
>>>
"""