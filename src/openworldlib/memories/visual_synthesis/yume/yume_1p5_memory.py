from typing import Optional, List, Dict, Any
from PIL import Image

from ...base_memory import BaseMemory


class Yume1p5Memory(BaseMemory):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.storage: List[Dict[str, Any]] = []
        self.all_frames: List[Image.Image] = []

        self.ref_images = None
        self.ref_videos = None
        self.n_generated_segments: int = 0

    def has_context(self) -> bool:
        return (self.ref_images is not None) or (self.ref_videos is not None)

    def record(self, data, record_frames: bool = True, as_context: bool = False, **kwargs):

        current_image: Optional[Image.Image] = None

        if isinstance(data, Image.Image):
            current_image = data
        elif isinstance(data, list):
            if len(data) > 0 and isinstance(data[-1], Image.Image):
                current_image = data[-1]
                if record_frames:
                    self.all_frames.extend(data)
            if (not as_context) and len(data) > 0:
                self.n_generated_segments += 1
        else:
            raise TypeError(f"Unsupported data type for record(): {type(data)}")

        visual_context = kwargs.get("visual_context", None)
        if visual_context is not None:
            if "ref_images" in visual_context:
                if (visual_context["ref_images"] is not None) or as_context:
                    self.ref_images = visual_context["ref_images"]
            if "ref_videos" in visual_context:
                if (visual_context["ref_videos"] is not None) or as_context:
                    self.ref_videos = visual_context["ref_videos"]

        self.storage.append(
            {
                "content": current_image,
                "type": "image",
                "timestamp": len(self.all_frames),
                "metadata": {"n_generated_segments": self.n_generated_segments},
            }
        )

    def select(self, **kwargs) -> Optional[Image.Image]:
        if len(self.storage) == 0:
            return None
        return self.storage[-1]["content"]

    def select_context(self) -> Optional[Dict[str, Any]]:
        if not self.has_context():
            return None
        return {
            "ref_images": self.ref_images,
            "ref_videos": self.ref_videos,
        }

    def manage(self, action: str = "reset", **kwargs):
        if action == "reset":
            self.storage = []
            self.all_frames = []
            self.ref_images = None
            self.ref_videos = None
            self.n_generated_segments = 0
