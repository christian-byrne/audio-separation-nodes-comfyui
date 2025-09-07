from .src.separation import AudioSeparation
from .src.tempo_match import TempoMatch
from .src.crop import AudioCrop
from .src.combine import AudioCombine
from .src.combine_video_with_audio import AudioVideoCombine
from .src.time_shift import TimeShift
from .src.get_tempo import GetTempo
from .src.split_equal import AudioSplitEqual, AudioSplitEqualList
from .src.list_get import AudioListGet
from .src.stitch import AudioStitch


NODE_CLASS_MAPPINGS = {
    "AudioSeparation": AudioSeparation,
    "AudioCrop": AudioCrop,
    "AudioCombine": AudioCombine,
    "AudioTempoMatch": TempoMatch,
    "AudioVideoCombine": AudioVideoCombine,
    "AudioSpeedShift": TimeShift,
    "AudioGetTempo": GetTempo,
    "AudioSplitEqual": AudioSplitEqual,
    "AudioSplitEqualList": AudioSplitEqualList,
    "AudioListGet": AudioListGet,
    "AudioStitch": AudioStitch,
}
