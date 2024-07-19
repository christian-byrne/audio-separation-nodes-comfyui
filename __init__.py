from .src.separation import AudioSeparation
from .src.tempo_match import TempoMatch
from .src.crop import AudioCrop
from .src.combine import AudioCombine
from .src.combine_video_with_audio import AudioVideoCombine
from .src.time_shift import TimeShift
from .src.get_tempo import GetTempo


NODE_CLASS_MAPPINGS = {
    "AudioSeparation": AudioSeparation,
    "AudioCrop": AudioCrop,
    "AudioCombine": AudioCombine,
    "AudioTempoMatch": TempoMatch,
    "AudioVideoCombine": AudioVideoCombine,
    "AudioSpeedShift": TimeShift,
    "AudioGetTempo": GetTempo,
}