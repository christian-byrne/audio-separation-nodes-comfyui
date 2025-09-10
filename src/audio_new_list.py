from typing import List, Tuple, Optional, Any

from ._types import AUDIO


class AudioNewList:
    @classmethod
    def INPUT_TYPES(cls):
        # Two required audio inputs and a dynamic input count controller.
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 2, "max": 1000, "step": 1, "tooltip": "Number of AUDIO inputs to expose. Click 'Update inputs' to apply."}),
                "audio_1": ("AUDIO",),
                "audio_2": ("AUDIO",),
            },
            "optional": {
                "ignore_empty": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If True, skip unconnected inputs. If False, raise if any requested input is missing.",
                    },
                )
            },
        }

    FUNCTION = "main"
    RETURN_TYPES = ("AUDIO_LIST",)
    RETURN_NAMES = ("List",)
    CATEGORY = "audio"
    DESCRIPTION = "Create a new AUDIO_LIST from multiple AUDIO inputs in the provided order. Use inputcount + Update inputs to adjust ports."

    def main(
        self,
        inputcount: int,
        ignore_empty: bool = True,
        **kwargs: Any,
    ) -> Tuple[List[AUDIO]]:
        # Clamp to sensible bounds
        try:
            n = int(inputcount)
        except Exception:
            n = 2
        n = max(2, min(1000, n))

        # Collect in order: audio_1 .. audio_n
        out: List[AUDIO] = []

        for idx in range(1, n + 1):
            key = f"audio_{idx}"
            val = kwargs.get(key)
            if val is None:
                if ignore_empty:
                    continue
                raise ValueError(f"AudioNewList: Missing required input '{key}' while 'ignore_empty' is False.")
            out.append(val)

        if len(out) < 2:
            raise ValueError("AudioNewList: Need at least two AUDIO inputs to form a list.")

        return (out,)
