import os
import platform
import tempfile
from pathlib import Path

import torch
import torchaudio

try:
    # moviepy<=1.0.3
    from moviepy.editor import VideoFileClip, AudioFileClip
except ImportError:
    # moviepy>=2.0.0 (Nov. 2024)
    from moviepy import VideoFileClip, AudioFileClip


from typing import Tuple
from ._types import AUDIO

import folder_paths


class AudioVideoCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "video_path": (
                    "STRING",
                    {
                        "default": "/path/to/video.mp4",
                        "tooltip": "The absolute file path to the video file to which the audio will be added.",
                    },
                ),
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
                        "default": "1:00",
                        "tooltip": "The video will be trimmed to end at this time. The format is MM:SS.",
                    },
                ),
                "auto_open": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "Auto open after combining",
                        "description": "Don't auto open after combining",
                        "tooltip": "Whether to automatically open the combined video with the default video player after processing.",
                    },
                ),
            },
        }

    FUNCTION = "main"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_video_path",)
    CATEGORY = "audio"
    OUTPUT_NODE = True
    OUTPUT_TOOLTIPS = ("The path to the output video.",)
    DESCRIPTION = "Replace the audio of a video with a new audio track."

    def main(
        self,
        audio: AUDIO,
        video_path: str = "/path/to/video.mp4",
        video_start_time: str = "0:00",
        video_end_time: str = "1:00",
        auto_open: bool = False,
    ) -> Tuple[str]:

        waveform: torch.Tensor = audio["waveform"]
        sample_rate: int = audio["sample_rate"]
        input_path = Path(video_path)
        if not input_path.exists():
            raise FileNotFoundError(
                f"AudioVideoCombine: Video file not found: {video_path}"
            )

        # Assume that no ":" in input means that the user is trying to specify seconds
        if ":" not in video_start_time:
            video_start_time = f"00:{video_start_time}"
        if ":" not in video_end_time:
            video_end_time = f"00:{video_end_time}"

        start_seconds_time = 60 * int(video_start_time.split(":")[0]) + int(
            video_start_time.split(":")[1]
        )
        end_seconds_time = 60 * int(video_end_time.split(":")[0]) + int(
            video_end_time.split(":")[1]
        )

        if start_seconds_time > end_seconds_time:
            raise ValueError(
                "AudioVideoCombine: Start time must be less than end time. Start time cannot be after video ends."
            )

        output_dir = Path(folder_paths.get_output_directory())
        filename = input_path.stem
        new_filename = f"{filename}_0_combined.mp4"
        index = 0
        while new_filename in [f.name for f in output_dir.iterdir()]:
            index += 1
            new_filename = f"{filename}_{index}_combined.mp4"

        new_filepath = str(output_dir / new_filename)

        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            torchaudio.save(f.name, waveform.squeeze(0), sample_rate=sample_rate)
            video = VideoFileClip(str(video_path), audio=False)
            audio = AudioFileClip(f.name)

            try:
                # moviepy<=1.0.3
                video = video.subclip(start_seconds_time, end_seconds_time)
                video = video.set_audio(audio)
            except AttributeError:
                # moviepy>=2.0.0 (Nov. 2024)
                video = video.subclipped(start_seconds_time, end_seconds_time)
                video = video.with_audio(audio)

            video.write_videofile(new_filepath, codec="libx264", audio_codec="aac")

        new_filepath = os.path.normpath(new_filepath)
        if auto_open:
            if platform.system() == "Darwin":
                os.system(f'open "{new_filepath}"')
            elif platform.system() == "Windows":
                os.system(f'start "{new_filepath}"')
            else:
                os.system(f'xdg-open "{new_filepath}"')

        return (str(new_filepath),)
