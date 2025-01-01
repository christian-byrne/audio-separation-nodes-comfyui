
![worklow picture](./wiki/pics/Selection_016.png)


https://github.com/user-attachments/assets/c5cf20de-a17f-438d-81ac-0c392af669cf



# Examples

#### *`Separating Voices in a Video`*

<details>

<summary> &nbsp; Show </summary>

> [!NOTE] 
> 
> In order to load videos into the LoadAudio Node, change [this line](https://github.com/comfyanonymous/ComfyUI/blob/faa57430b0ff882275b1afcf6610e8e9f8a5929b/comfy_extras/nodes_audio.py#L185) in your Comfy install to include the video's extension (e.g., `.mp4`)

[workflow.json](./wiki/workflows/isolate-vocals-video.json)

https://github.com/user-attachments/assets/c5af418e-7137-4c36-b86e-3352cf558ea8

</details>







#### *`Replacing BGM with StableAudio-Generated BGM`*

<details>
  
<summary> &nbsp; Show </summary>


> [!NOTE] 
> 
> In order to load videos into the LoadAudio Node, change [this line](https://github.com/comfyanonymous/ComfyUI/blob/faa57430b0ff882275b1afcf6610e8e9f8a5929b/comfy_extras/nodes_audio.py#L185) in your Comfy install to include the video's extension (e.g., `.mp4`)

You can use this to replace copyrighted BGM in a video with new BGM. You can set the denoise low, so that the new BGM is still stimilar to the original.

[workflow json](./wiki/workflows/replace-bgm.json)

https://github.com/user-attachments/assets/a7d5656b-5f8b-439a-936f-6ebb6a0d538a

</details>


#### *`Remixing Songs with StableAudio`*


<details>

<summary> &nbsp; Show </summary>

- [workflow json](./wiki/workflows/remix-songs.json)
- [example output (audio file) with embedded workflow](./wiki/examples/ComfyUI_temp_iaepj_00001_.flac)
- [example output (audio file) with embedded workflow](./wiki/examples/ComfyUI_00002_.flac)

</details>


#### *`Separating Song Vocals`*

<details>

<summary> &nbsp; Show </summary>

[workflow.json](./wiki/workflows/isolate-vocals-song.json)

https://github.com/user-attachments/assets/c5cf20de-a17f-438d-81ac-0c392af669cf

</details>




#### *`Extracting Instrumentals from Songs`*


<details>

<summary> &nbsp; Show </summary>

- [workflow json](./wiki/workflows/extract-instrumental.json)

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
2. `cd` into the cloned repository
3. `pip install -r requirements.txt`
