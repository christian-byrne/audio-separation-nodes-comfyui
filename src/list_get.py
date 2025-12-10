from typing import Tuple, List, Any


class AudioListGet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_list": ("AUDIO_LIST",),
                "index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10_000,
                        "step": 1,
                        "tooltip": "Zero-based index into the audio list.",
                    },
                ),
            },
            "optional": {
                "out_of_range": (
                    ["error", "clamp", "wrap"],
                    {
                        "default": "clamp",
                        "tooltip": "Behavior when index is out of range: error, clamp to range, or wrap modulo length.",
                    },
                ),
            },
        }

    FUNCTION = "main"
    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("Audio", "Length")
    CATEGORY = "audio"
    DESCRIPTION = "Get a single AUDIO from an AUDIO_LIST by index."

    def main(self, audio_list: List[Any], index: int = 0, out_of_range: str = "clamp") -> Tuple[Any, int]:
        if not isinstance(audio_list, list):
            raise TypeError("AudioListGet: 'audio_list' must be a list.")
        if len(audio_list) == 0:
            raise ValueError("AudioListGet: 'audio_list' is empty.")

        n = len(audio_list)
        i = int(index)

        if i < 0 or i >= n:
            if out_of_range == "error":
                raise IndexError(f"AudioListGet: index {i} out of range [0, {n-1}].")
            elif out_of_range == "wrap":
                i = i % n
            else:  # clamp
                i = 0 if i < 0 else n - 1

        return (audio_list[i], n)
