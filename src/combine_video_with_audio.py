import tempfile
from pathlib import Path

import torch
import torchaudio
from moviepy.editor import VideoFileClip, AudioFileClip

from typing import Tuple
from ._types import AUDIO

import folder_paths

class AudioVideoCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "video_path": ("STRING", {"default": "/path/to/video.mp4"}),
            },
            "optional": {
                "video_start_time": (
                    "STRING",
                    {
                        "default": "0:00",
                    },
                ),
                "video_end_time": (
                    "STRING",
                    {
                        "default": "1:00",
                    },
                ),
            },
        }

    FUNCTION = "main"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_video_path",)
    CATEGORY = "audio"
    OUTPUT_NODE = True

    def main(
        self,
        audio: AUDIO,
        video_path: str = "/path/to/video.mp4",
        video_start_time: str = "0:00",
        video_end_time: str = "1:00",
    ) -> Tuple[str]:

        waveform: torch.Tensor = audio["waveform"]
        sample_rate: int = audio["sample_rate"]
        input_path = Path(video_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

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

        assert (
            start_seconds_time < end_seconds_time
        ), "AudioVideoCombine: Start time must be less than end time. Start time cannot be after video ends."

        filename = input_path.stem
        new_filename = f"{filename}_0_combined.mp4"
        output_dir = Path(folder_paths.get_output_directory())
        index = 0
        while new_filename in [f.name for f in output_dir.iterdir()]:
            index += 1
            new_filename = f"{filename}_{index}_combined.mp4"

        new_filepath = output_dir / new_filename

        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            torchaudio.save(f.name, waveform.squeeze(0), sample_rate=sample_rate)
            video = VideoFileClip(str(video_path), audio=False)
            video = video.subclip(start_seconds_time, end_seconds_time)
            audio = AudioFileClip(f.name)
            video = video.set_audio(audio)
            video.write_videofile(str(new_filepath), codec="libx264", audio_codec="aac")

        return (str(new_filepath),)
