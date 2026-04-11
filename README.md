![worklow picture](https://github.com/christian-byrne/audio-separation-nodes-comfyui/blob/demo-files/wiki/pics/Selection_016.png?raw=true)

https://github.com/user-attachments/assets/c5cf20de-a17f-438d-81ac-0c392af669cf

# Nodes

| Node | Description |
|------|-------------|
| **Audio Separation** | Separate audio into four stems (bass, drums, other, vocals) using [Hybrid Demucs](https://pytorch.org/audio/stable/tutorials/hybrid_demucs_tutorial.html). |
| **Audio Combine** | Combine two audio tracks by overlaying their waveforms (add, subtract, multiply, divide, mean). |
| **Audio Crop** | Crop (trim) audio to a specific start and end time. |
| **Audio Tempo Match** | Match the tempo of two audio tracks by time-stretching both to their average BPM. |
| **Audio Speed Shift** | Time-stretch or time-compress audio by a given rate. |
| **Audio Get Tempo** | Get the tempo (BPM) of audio using onset detection. |
| **Audio Video Combine** | Replace the audio of a VIDEO input with a new audio track. |

# Examples

#### _`Separating Voices in a Video`_

<details>

<summary> &nbsp; Show </summary>

> [!NOTE]
>
> In order to load videos into the LoadAudio Node, change [this line](https://github.com/comfyanonymous/ComfyUI/blob/faa57430b0ff882275b1afcf6610e8e9f8a5929b/comfy_extras/nodes_audio.py#L185) in your Comfy install to include the video's extension (e.g., `.mp4`)

[workflow.json](./example_workflows/Isolate%20Vocals%20from%20Video.json)

https://github.com/user-attachments/assets/c5af418e-7137-4c36-b86e-3352cf558ea8

</details>

#### _`Replacing BGM with StableAudio-Generated BGM`_

<details>
  
<summary> &nbsp; Show </summary>

> [!NOTE]
>
> In order to load videos into the LoadAudio Node, change [this line](https://github.com/comfyanonymous/ComfyUI/blob/faa57430b0ff882275b1afcf6610e8e9f8a5929b/comfy_extras/nodes_audio.py#L185) in your Comfy install to include the video's extension (e.g., `.mp4`)

You can use this to replace copyrighted BGM in a video with new BGM. You can set the denoise low, so that the new BGM is still stimilar to the original.

[workflow json](./example_workflows/Replace%20BGM%20with%20Stable-Audio-Generated%20Music.json)

https://github.com/user-attachments/assets/a7d5656b-5f8b-439a-936f-6ebb6a0d538a

</details>

#### _`Remixing Songs with StableAudio`_

<details>

<summary> &nbsp; Show </summary>

- [workflow json](./example_workflows/Remix%20Song.json)
- [example output (audio file) with embedded workflow](https://github.com/christian-byrne/audio-separation-nodes-comfyui/raw/refs/heads/demo-files/wiki/examples/ComfyUI_temp_iaepj_00001_.flac)
- [example output (audio file) with embedded workflow](https://github.com/christian-byrne/audio-separation-nodes-comfyui/raw/refs/heads/demo-files/wiki/examples/ComfyUI_00002_.flac)

</details>

#### _`Separating Song Vocals`_

<details>

<summary> &nbsp; Show </summary>

[workflow.json](./example_workflows/Isolate%20Vocals%20from%20Audio.json)

https://github.com/user-attachments/assets/c5cf20de-a17f-438d-81ac-0c392af669cf

</details>

#### _`Extracting Instrumentals from Songs`_

<details>

<summary> &nbsp; Show </summary>

- [workflow json](./example_workflows/Extract%20Instrumental%20from%20Song.json)

</details>

&nbsp;

# Stem Mapping

The **Audio Separation** node uses [Hybrid Demucs](https://pytorch.org/audio/stable/tutorials/hybrid_demucs_tutorial.html) to split audio into four stems:

| Output | Contains |
|--------|----------|
| **Bass** | Bass guitar, sub-bass, low-frequency instruments |
| **Drums** | Drums, percussion, hi-hats |
| **Other** | Everything else — guitars, keyboards, synths, strings, etc. |
| **Vocals** | Singing, speech, vocal harmonies |

> **Looking for a specific instrument like guitar?** Guitar is included in the
> **Other** stem. To isolate guitar, separate first, then use the **Audio Combine**
> node to subtract unwanted elements or further process the "Other" output.

# Requirements

```
librosa>=0.10.2,<1
torchaudio>=2.3.0
numpy
moviepy
```

# Installation

1. If you run ComfyUI inside of a virtual environment, make sure it is activated
1. `git clone` this repository in `ComfyUI/custom_nodes` folder
1. `cd` into the cloned repository
1. `pip install -r requirements.txt`

# Troubleshooting

<details>
<summary><b>BadZipFile / "failed finding central directory"</b></summary>

This error means the Hybrid Demucs model checkpoint was corrupted during download.
Delete the cached file and restart ComfyUI to trigger a fresh download:

```bash
# Default location (Linux/macOS)
rm ~/.cache/torch/hub/checkpoints/*.th

# Windows
del %USERPROFILE%\.cache\torch\hub\checkpoints\*.th
```

See [#21](https://github.com/christian-byrne/audio-separation-nodes-comfyui/issues/21).

</details>

<details>
<summary><b>ConnectionResetError on Windows</b></summary>

```
Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)
ConnectionResetError: [WinError 10054]
```

This is harmless Windows asyncio noise — it does not affect audio separation results.
The error comes from Python's `ProactorEventLoop` closing connections and can be
safely ignored. See [#9](https://github.com/christian-byrne/audio-separation-nodes-comfyui/issues/9).

</details>
