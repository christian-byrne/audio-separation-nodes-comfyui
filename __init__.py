from .separation import AudioSeparation
from .tempo_match import TempoMatch
from .crop import AudioCrop
from .generation import AudioGeneration
from .combine import AudioCombine


NODE_CLASS_MAPPINGS = {
    "AudioSeparation": AudioSeparation,
    "AudioCrop": AudioCrop,
    "AudioGeneration": AudioGeneration,
    "AudioCombine": AudioCombine,
    "AudioTempoMatch": TempoMatch,

}
