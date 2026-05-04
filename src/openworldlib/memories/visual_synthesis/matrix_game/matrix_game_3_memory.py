from ...base_memory import BaseMemory
from typing import Optional, Any, Dict
from PIL import Image


class MatrixGame3Memory(BaseMemory):
    """
    Memory module for Matrix-Game-3.

    Stores the latest input image (for stream/i2v compatibility) and
    clip-level state for v2v (video continuation). The clip_state dict
    holds latent-level state returned by generate_clip, including
    all_latents_list, extrinsics_all, img_cond, vae_cache, generator, etc.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.storage = []
        self._clip_state: Optional[Dict[str, Any]] = None
        self._last_pose: Optional[Any] = None

    def record(self, data, **kwargs):
        if isinstance(data, Image.Image):
            self.storage.append(
                {
                    "content": data,
                    "type": "image",
                    "timestamp": len(self.storage),
                    "metadata": {},
                }
            )
        elif isinstance(data, list):
            for image in data:
                self.storage.append(
                    {
                        "content": image,
                        "type": "image",
                        "timestamp": len(self.storage),
                        "metadata": {},
                    }
                )

    def select(self, **kwargs) -> Optional[Image.Image]:
        if not self.storage:
            return None
        return self.storage[-1]["content"]

    def has_continuation_state(self) -> bool:
        return self._clip_state is not None

    def get_clip_state(self) -> Optional[Dict[str, Any]]:
        if self._clip_state is not None:
            state = dict(self._clip_state)
            if self._last_pose is not None:
                state["last_pose"] = self._last_pose
            return state
        return None

    def set_clip_state(self, clip_state: Dict[str, Any]):
        self._clip_state = clip_state
        if "last_pose" in clip_state:
            self._last_pose = clip_state["last_pose"]

    def manage(self, action: str = "reset", **kwargs):
        if action == "reset":
            self.storage = []
            self._clip_state = None
            self._last_pose = None

