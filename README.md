
![worklow picture](./wiki/pics/Selection_016.png)


![Demo video - isolating vocals song](./wiki/videos/isolate-vocals-song.mp4)

![Demo video - isolating vocals movie scene](https://github.com/christian-byrne/audio-separation-nodes-comfyui/blob/master/wiki/videos/isolate-vocals-video.mp4)

# Workflow Examples

#### *`Separating Voices in a Video`*

>
>
> <details>
>
> <summary> &nbsp; Show </summary>
>
> - [workflow.json](./wiki/workflows/isolate-vocals-video.json)
> - Example output:
>     ![example output](./wiki/videos/isolate-vocals-video.mp4)
> 
> </details>
> 


#### *`Separating Song Vocals`*

>
>
> <details>
>
> <summary> &nbsp; Show </summary>
>
> - [workflow.json](./wiki/workflows/isolate-vocals-song.json)
> - Example:
>     ![example output](./wiki/videos/isolate-vocals-song.mp4)
> 
>
> </details>
> 


#### *`Replacing BGM with Generated BGM`*

>
>
> <details>
>
> <summary> &nbsp; Show </summary>
>
> &nbsp; *For example, to replace copyrighted BGM with new music that has the same mood*.
>
> - [workflow json](./wiki/workflows/replace-bgm.json)
> - Example output:
>     ![example output](./wiki/videos/bgm-replace.mp4)
>   - *NOTE*: In order to load videos into the LoadAudio Node, change [this line](https://github.com/comfyanonymous/ComfyUI/blob/faa57430b0ff882275b1afcf6610e8e9f8a5929b/comfy_extras/nodes_audio.py#L185) in your comfy install to include the `.ext` (e.g., `.mp4`)
>
> </details>


#### *`Remixing Songs with StableAudio`*

>
>
> <details>
>
> <summary> &nbsp; Show </summary>
>
> - [workflow json](./wiki/workflows/remix-songs.json)
> - [example output (audio file) with embedded workflow](./wiki/examples/ComfyUI_temp_ksudt_00002_.flac)
> - [example output (audio file) with embedded workflow](./wiki/examples/ComfyUI_00002_.flac)
>
> </details>


&nbsp;

# Requirements

```m
librosa==0.10.2
torchaudio>=2.3.0
numpy
moviepy
```

# Installation

1. `git clone` this repository in `ComfyUI/custom_nodes` folder
2. `cd` into the cloned repository
3. `pip install -r requirements.txt`
