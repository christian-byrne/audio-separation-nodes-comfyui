try:  # pragma: no cover - Comfy runtime provides dependencies
    from .src.separation import AudioSeparation
except Exception:  # pragma: no cover
    try:
        from src.separation import AudioSeparation
    except Exception:  # pragma: no cover
        AudioSeparation = None

try:
    from .src.tempo_match import TempoMatch
except Exception:
    try:
        from src.tempo_match import TempoMatch
    except Exception:
        TempoMatch = None

try:
    from .src.crop import AudioCrop
except Exception:
    try:
        from src.crop import AudioCrop
    except Exception:
        AudioCrop = None

try:
    from .src.combine import AudioCombine
except Exception:
    try:
        from src.combine import AudioCombine
    except Exception:
        AudioCombine = None

try:
    from .src.combine_video_with_audio import AudioVideoCombine
except Exception:
    try:
        from src.combine_video_with_audio import AudioVideoCombine
    except Exception:
        AudioVideoCombine = None

try:
    from .src.time_shift import TimeShift
except Exception:
    try:
        from src.time_shift import TimeShift
    except Exception:
        TimeShift = None

try:
    from .src.get_tempo import GetTempo
except Exception:
    try:
        from src.get_tempo import GetTempo
    except Exception:
        GetTempo = None


NODE_CLASS_MAPPINGS = {}
if AudioSeparation:
    NODE_CLASS_MAPPINGS["AudioSeparation"] = AudioSeparation
if AudioCrop:
    NODE_CLASS_MAPPINGS["AudioCrop"] = AudioCrop
if AudioCombine:
    NODE_CLASS_MAPPINGS["AudioCombine"] = AudioCombine
if TempoMatch:
    NODE_CLASS_MAPPINGS["AudioTempoMatch"] = TempoMatch
if AudioVideoCombine:
    NODE_CLASS_MAPPINGS["AudioVideoCombine"] = AudioVideoCombine
if TimeShift:
    NODE_CLASS_MAPPINGS["AudioSpeedShift"] = TimeShift
if GetTempo:
    NODE_CLASS_MAPPINGS["AudioGetTempo"] = GetTempo
