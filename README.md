![worklow picture](https://github.com/christian-byrne/audio-separation-nodes-comfyui/blob/demo-files/wiki/pics/Selection_016.png?raw=true)

https://github.com/user-attachments/assets/c5cf20de-a17f-438d-81ac-0c392af669cf

# Examples

#### _`Separating Voices in a Video`_

<details>

<summary> &nbsp; Show </summary>

> [!NOTE]
>
> In order to load videos into the LoadAudio Node, change [this line](https://github.com/comfyanonymous/ComfyUI/blob/faa57430b0ff882275b1afcf6610e8e9f8a5929b/comfy_extras/nodes_audio.py#L185) in your Comfy install to include the video's extension (e.g., `.mp4`)

[workflow.json](./example_workflows/isolate-vocals-video.json)

https://github.com/user-attachments/assets/c5af418e-7137-4c36-b86e-3352cf558ea8

</details>

#### _`Replacing BGM with StableAudio-Generated BGM`_

<details>
  
<summary> &nbsp; Show </summary>

> [!NOTE]
>
> In order to load videos into the LoadAudio Node, change [this line](https://github.com/comfyanonymous/ComfyUI/blob/faa57430b0ff882275b1afcf6610e8e9f8a5929b/comfy_extras/nodes_audio.py#L185) in your Comfy install to include the video's extension (e.g., `.mp4`)

You can use this to replace copyrighted BGM in a video with new BGM. You can set the denoise low, so that the new BGM is still stimilar to the original.

[workflow json](./example_workflows/replace-bgm.json)

https://github.com/user-attachments/assets/a7d5656b-5f8b-439a-936f-6ebb6a0d538a

</details>

#### _`Remixing Songs with StableAudio`_

<details>

<summary> &nbsp; Show </summary>

- [workflow json](./example_workflows/remix-songs.json)
- [example output (audio file) with embedded workflow](https://github.com/christian-byrne/audio-separation-nodes-comfyui/raw/refs/heads/demo-files/wiki/examples/ComfyUI_temp_iaepj_00001_.flac)
- [example output (audio file) with embedded workflow](https://github.com/christian-byrne/audio-separation-nodes-comfyui/raw/refs/heads/demo-files/wiki/examples/ComfyUI_00002_.flac)

</details>

#### _`Separating Song Vocals`_

<details>

<summary> &nbsp; Show </summary>

[workflow.json](./example_workflows/isolate-vocals-song.json)

https://github.com/user-attachments/assets/c5cf20de-a17f-438d-81ac-0c392af669cf

</details>

#### _`Extracting Instrumentals from Songs`_

<details>

<summary> &nbsp; Show </summary>

- [workflow json](./example_workflows/extract-instrumental.json)

</details>

&nbsp;

# Requirements

```m
librosa==0.10.2
torchaudio>=2.3.0
numpy
moviepy
```

# Installation

1. If you run ComfyUI inside of a virtual environment, make sure it is activated
1. `git clone` this repository in `ComfyUI/custom_nodes` folder
1. `cd` into the cloned repository
1. `pip install -r requirements.txt`
