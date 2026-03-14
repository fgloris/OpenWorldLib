from typing import Any, Dict, Optional, List

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

from .base_operator import BaseOperator
from ..synthesis.visual_generation.yume.yume import YUME_SIZE_CONFIGS


YUME_SUPPORTED_SIZES = tuple(YUME_SIZE_CONFIGS.keys())


class YumeOperator(BaseOperator):
    """Lightweight operator for YUME prompt/image preprocessing."""

    def __init__(self, operation_types=[]) -> None:
        super(YumeOperator, self).__init__()
        self.interaction_template = [
            "forward",
            "left",
            "right",
            "backward",
            "camera_l",
            "camera_r",
            "camera_up",
            "camera_down",
        ]
        self.interaction_template_init()

    def check_interaction(self, interaction):
        if interaction not in self.interaction_template:
            raise ValueError(f"{interaction} not in template")
        return True

    def get_interaction(self, interactions):
        if not isinstance(interactions, list):
            interactions = [interactions]
        for interaction in interactions:
            self.check_interaction(interaction)
        self.current_interaction.append(interactions)

    def process_interaction(self, **kwargs) -> Dict[str, Any]:
        INTERACTION_2_CAPTION_DICT = {
            "forward": "The camera pushes forward (W).",
            "backward": "The camera pulls back (S).",
            "left": "Camera turns left (←).",
            "right": "Camera turns right (→).",
            "camera_up": "Camera tilts up (↑).",
            "camera_down": "Camera tilts down (↓).",
            "camera_l": "The camera pans to the left (←).",
            "camera_r": "The camera pans to the right (→).",
        }

        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")
        now_interaction = self.current_interaction[-1]
        self.interaction_history.append(now_interaction)
        return [INTERACTION_2_CAPTION_DICT[act] for act in now_interaction]

    def process_perception(
        self,
        size: Optional[str] = None,
        images: Optional[Image.Image] = None,
        videos: Optional[List[Image.Image]] = None,
    ) -> Dict[str, Any]:

        assert size in YUME_SUPPORTED_SIZES, (
            f"Unsupported size: {size}. "
            f"Supported sizes for yume are: {YUME_SUPPORTED_SIZES}"
        )
        target_size = YUME_SIZE_CONFIGS[size]

        resized_images = None
        video_pixel_values = None

        if images is not None:
            if isinstance(images, Image.Image) and images.mode != "RGB":
                images = images.convert("RGB")
            images = np.array(images)
            if len(images.shape) == 2:
                images = np.stack((images,) * 3, axis=-1)
            elif images.shape[2] == 4:
                images = images[:, :, :3]

            images_tensor = torch.from_numpy(images).permute(2, 0, 1).float() / 255.0
            resized_images = F.interpolate(
                images_tensor.unsqueeze(0),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )[0]

        if videos:
            video_transform = transforms.ToTensor()
            normalized_frames = []
            for frame in videos:
                if isinstance(frame, Image.Image) and frame.mode != "RGB":
                    frame = frame.convert("RGB")
                normalized_frames.append(video_transform(frame))
            video_pixel_values = torch.stack(normalized_frames, dim=0)
            video_pixel_values = (
                torch.nn.functional.interpolate(
                    video_pixel_values.sub_(0.5).div_(0.5),
                    size=target_size,
                    mode="bicubic",
                )
            ).clamp_(-1, 1)

        return {
            "ref_images": resized_images if images is not None else None,
            "ref_videos": video_pixel_values if videos is not None else None,
        }
