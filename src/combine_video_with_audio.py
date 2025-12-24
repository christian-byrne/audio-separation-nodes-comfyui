import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Tuple

import torch
import torchaudio
from comfy_api.input_impl import VideoFromFile
from comfy_api.input.video_types import VideoInput


try:
    # moviepy<=1.0.3
    from moviepy.editor import VideoFileClip, AudioFileClip
except ImportError:
    # moviepy>=2.0.0 (Nov. 2024)
    from moviepy import VideoFileClip, AudioFileClip


from ._types import AUDIO
from .audio_video_logic import compute_trim_window

import folder_paths


class AudioVideoCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "video": ("VIDEO",),
            },
            "optional": {
                "video_start_time": (
                    "STRING",
                    {
                        "default": "0:00",
                        "tooltip": "The video will be trimmed to start at this time. The format is MM:SS.",
                    },
                ),
                "video_end_time": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The video will be trimmed to end at this time. Leave blank to use the full duration.",
                    },
                ),
            },
        }

    FUNCTION = "main"
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    CATEGORY = "audio"
    OUTPUT_TOOLTIPS = ("The combined video.",)
    DESCRIPTION = "Replace the audio of a VIDEO input with a new audio track."

    def main(
        self,
        audio: AUDIO,
        video: VideoInput,
        video_start_time: str = "0:00",
        video_end_time: str = "",
    ) -> Tuple[VideoFromFile]:
        waveform: torch.Tensor = audio["waveform"]
        sample_rate: int = audio["sample_rate"]
        temp_dir = Path(folder_paths.get_temp_directory())
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            video_duration = video.get_duration()
        except Exception:
            video_duration = None

        start_seconds_time, end_seconds_time = compute_trim_window(
            video_start_time,
            video_end_time,
            video_duration,
        )
        clip_duration = end_seconds_time - start_seconds_time

        temp_input_path: Optional[Path] = None
        source = video.get_stream_source()
        if isinstance(source, (str, os.PathLike)) and Path(source).exists():
            video_path = str(source)
        else:
            temp_input_path = (
                temp_dir / f"audio_video_combine_input_{uuid.uuid4().hex}.mp4"
            )
            video.save_to(str(temp_input_path))
            video_path = str(temp_input_path)

        output_path = temp_dir / f"audio_video_combine_{uuid.uuid4().hex}.mp4"

        target_samples = (
            int(round(clip_duration * sample_rate)) if clip_duration > 0 else 0
        )
        trimmed_waveform = waveform
        if target_samples > 0 and waveform.shape[-1] > target_samples:
            trimmed_waveform = waveform[..., :target_samples]

        temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio_file.close()
        try:
            torchaudio.save(
                temp_audio_file.name,
                trimmed_waveform.squeeze(0),
                sample_rate=sample_rate,
            )
            video = VideoFileClip(str(video_path), audio=False)
            audio = AudioFileClip(temp_audio_file.name)

            try:
                # moviepy<=1.0.3
                video = video.subclip(start_seconds_time, end_seconds_time)
                video = video.set_audio(audio)
            except AttributeError:
                # moviepy>=2.0.0 (Nov. 2024)
                video = video.subclipped(start_seconds_time, end_seconds_time)
                video = video.with_audio(audio)

            video.write_videofile(str(output_path), codec="libx264", audio_codec="aac")

            video.close()
            audio.close()
        finally:
            try:
                Path(temp_audio_file.name).unlink(missing_ok=True)
            except OSError:
                pass

        if temp_input_path and temp_input_path.exists():
            try:
                temp_input_path.unlink()
            except OSError:
                pass

        return (VideoFromFile(str(output_path)),)
