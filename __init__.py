from .separation import AudioSeparation
from .tempo_match import TempoMatch
from .crop import AudioCrop
from .combine import AudioCombine


NODE_CLASS_MAPPINGS = {
    "AudioSeparation": AudioSeparation,
    "AudioCrop": AudioCrop,
    "AudioCombine": AudioCombine,
    "AudioTempoMatch": TempoMatch,

}
