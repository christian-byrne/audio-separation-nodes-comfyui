
![worklow picture](./wiki/pics/Selection_016.png)

## Examples

- Isolating/Separating Vocals
- Remixing Songs
  - [workflow json](./wiki/workflows/remix-songs.json)
  - [audio file with workflow](./wiki/examples/ComfyUI_temp_ksudt_00002_.flac)
  - [audio file with workflow](./wiki/examples/ComfyUI_00002_.flac)
- Replacing BGM of a video (e.g., to remove copyright BGM)
  - [workflow json](./wiki/workflows/replace-bgm.json)
  - Example: [before video](./wiki/videos/westworld.webm) | [after video](./wiki/videos/westworld-bgm-replaced.mp4)
  - In order to load videos into the LoadAudio Node, change [this line](https://github.com/comfyanonymous/ComfyUI/blob/faa57430b0ff882275b1afcf6610e8e9f8a5929b/comfy_extras/nodes_audio.py#L185) in your comfy install

## Requirements

```m
librosa==0.10.2
torchaudio>=2.3.0
numpy
moviepy
```

## Installation

1. `git clone` this repository in `ComfyUI/custom_nodes` folder
2. `cd` into the cloned repository
3. `pip install -r requirements.txt`