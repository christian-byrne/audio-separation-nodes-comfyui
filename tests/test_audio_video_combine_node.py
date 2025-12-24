from __future__ import annotations

from pathlib import Path
import importlib
import sys
import types

import numpy as np
import pytest


class _DummyVideoFromFile:
    def __init__(self, path: str):
        self._path = path

    def get_stream_source(self):
        return self._path


if "torch" not in sys.modules:
    sys.modules["torch"] = types.SimpleNamespace(Tensor=object)
if "torchaudio" not in sys.modules:
    sys.modules["torchaudio"] = types.SimpleNamespace(save=lambda *_, **__: None)

if "comfy_api" not in sys.modules:
    comfy_api = types.ModuleType("comfy_api")
    comfy_api.__path__ = []
    sys.modules["comfy_api"] = comfy_api
else:
    comfy_api = sys.modules["comfy_api"]

input_impl_module = types.ModuleType("comfy_api.input_impl")
input_impl_module.VideoFromFile = _DummyVideoFromFile
sys.modules["comfy_api.input_impl"] = input_impl_module
setattr(comfy_api, "input_impl", input_impl_module)

input_module = types.ModuleType("comfy_api.input")
input_module.__path__ = []
video_types_module = types.ModuleType("comfy_api.input.video_types")
video_types_module.VideoInput = object
sys.modules["comfy_api.input"] = input_module
sys.modules["comfy_api.input.video_types"] = video_types_module
setattr(input_module, "video_types", video_types_module)
setattr(comfy_api, "input", input_module)

moviepy_module = types.ModuleType("moviepy")
editor_module = types.ModuleType("moviepy.editor")
moviepy_module.editor = editor_module
editor_module.VideoFileClip = object
editor_module.AudioFileClip = object
moviepy_module.VideoFileClip = object
moviepy_module.AudioFileClip = object
sys.modules.setdefault("moviepy", moviepy_module)
sys.modules.setdefault("moviepy.editor", editor_module)

folder_paths_module = types.ModuleType("folder_paths")
folder_paths_module.get_temp_directory = lambda: Path.cwd() / "temp"
sys.modules.setdefault("folder_paths", folder_paths_module)

module = importlib.import_module("src.combine_video_with_audio")


class StubVideo:
    def __init__(self, source_path: str, duration: float):
        self._source_path = source_path
        self._duration = duration

    def get_stream_source(self):
        return self._source_path

    def get_duration(self):
        return self._duration

    def save_to(self, path: str):
        Path(path).write_bytes(b"")


@pytest.fixture
def dummy_audio():
    return {
        "waveform": np.zeros((1, 1, 1000), dtype=float),
        "sample_rate": 100,
    }


def test_audio_video_combine_replaces_audio(tmp_path, monkeypatch, dummy_audio):
    calls = {}

    class DummyVideoClip:
        def __init__(self, path, audio=False):
            calls["video_init"] = (path, audio)

        def subclip(self, start, end):
            calls["subclip"] = (start, end)
            return self

        def set_audio(self, audio_clip):
            calls["set_audio"] = audio_clip
            return self

        def write_videofile(self, path, codec, audio_codec):
            Path(path).write_bytes(b"video")
            calls["write"] = (path, codec, audio_codec)

        def close(self):
            calls["video_closed"] = True

    class DummyAudioClip:
        def __init__(self, path):
            calls["audio_init"] = path

        def close(self):
            calls["audio_closed"] = True

    monkeypatch.setattr(module, "VideoFileClip", DummyVideoClip)
    monkeypatch.setattr(module, "AudioFileClip", DummyAudioClip)
    monkeypatch.setattr(module.folder_paths, "get_temp_directory", lambda: tmp_path)

    def fake_save(filename, waveform, sample_rate):
        Path(filename).write_bytes(b"wav")
        calls["audio_samples"] = waveform.shape[-1]
        calls["audio_sample_rate"] = sample_rate

    monkeypatch.setattr(module.torchaudio, "save", fake_save)

    stub_video = StubVideo(source_path=str(tmp_path / "source.mp4"), duration=12.0)

    combine = module.AudioVideoCombine()
    (result,) = combine.main(dummy_audio, stub_video, "0:05", "0:08")

    output_path = result.get_stream_source()
    assert Path(output_path).exists()
    assert calls["subclip"] == (5.0, 8.0)
    assert calls["write"][1:] == ("libx264", "aac")
    assert calls["audio_sample_rate"] == 100
    assert calls["audio_samples"] == 300
