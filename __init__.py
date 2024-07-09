from .src.separation import AudioSeparation
from .src.tempo_match import TempoMatch
from .src.crop import AudioCrop
from .src.combine import AudioCombine
from .src.combine_video_with_audio import AudioVideoCombine


NODE_CLASS_MAPPINGS = {
    "AudioSeparation": AudioSeparation,
    "AudioCrop": AudioCrop,
    "AudioCombine": AudioCombine,
    "AudioTempoMatch": TempoMatch,
    "AudioVideoCombine": AudioVideoCombine,
}
