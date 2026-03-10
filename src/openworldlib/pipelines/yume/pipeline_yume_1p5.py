from typing import Any, Dict, Optional, List, Union, Tuple
from PIL import Image
import numpy as np
import torch
from ...operators.yume_1p5_operator import Yume1p5Operator
from ...synthesis.visual_generation.yume.yume_1p5_synthesis import Yume1p5Synthesis
from ...memories.visual_synthesis.yume.yume_1p5_memory import Yume1p5Memory


class Yume1p5Pipeline:

    def __init__(
        self,
        synthesis_model: Optional[Yume1p5Synthesis] = None,
        operators: Optional[Yume1p5Operator] = None,
        memory_module: Optional[Any] = None,
        device: str = "cuda",
        weight_dtype = torch.bfloat16,
    ) -> None:
        
        self.operators = operators
        self.synthesis_model = synthesis_model
        self.memory_module = memory_module
        self.device = device
        self.weight_dtype = weight_dtype

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        device,
        weight_dtype,
        fsdp
    ) -> "Yume1p5Pipeline":
        if model_path is None:
            synthesis_model_path = "stdstu123/Yume-5B-720P"
        else:
            synthesis_model_path = model_path

        print(f"Loading Yume synthesis model from {synthesis_model_path}...")
        synthesis_model = Yume1p5Synthesis.from_pretrained(
            pretrained_model_path=synthesis_model_path,
            device=device,
            weight_dtype=weight_dtype,
            fsdp=fsdp
        )
        operators = Yume1p5Operator()
        memory_module = Yume1p5Memory()

        pipeline = cls(
            operators=operators,
            synthesis_model=synthesis_model,
            memory_module=memory_module,
            device=device,
            weight_dtype=weight_dtype,
        )
        return pipeline

    def _dist_rank(self) -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0

    def _to_pil_frames(self, video: Any) -> List[Image.Image]:
        def _frame_to_pil(frame: np.ndarray) -> Image.Image:
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            if frame.ndim != 3:
                raise ValueError(f"Unsupported frame shape: {frame.shape}")
            if frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            if frame.shape[-1] == 4:
                frame = frame[:, :, :3]

            if frame.dtype != np.uint8:
                if np.issubdtype(frame.dtype, np.floating):
                    if frame.min() < 0:
                        frame = (frame + 1.0) / 2.0
                    if frame.max() <= 1.0:
                        frame = np.clip(frame, 0.0, 1.0)
                        frame = (frame * 255.0).astype(np.uint8)
                    else:
                        frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
                else:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
            return Image.fromarray(frame)

        if isinstance(video, list):
            if len(video) == 0:
                return []
            if isinstance(video[0], Image.Image):
                return video
            if isinstance(video[0], np.ndarray):
                return [_frame_to_pil(frame) for frame in video]
            raise TypeError(f"Unsupported frame type in list: {type(video[0])}")

        if isinstance(video, np.ndarray):
            if video.ndim != 4:
                raise ValueError(f"Expected video ndarray shape (T,H,W,C), got {video.shape}")
            return [_frame_to_pil(frame) for frame in video]

        if isinstance(video, torch.Tensor):
            v = video.detach().cpu()
            if v.ndim != 4:
                raise ValueError(f"Expected video tensor ndim=4, got shape {tuple(v.shape)}")

            if v.shape[-1] in (1, 3, 4):
                arr = v.numpy()  # (T,H,W,C)
            elif v.shape[1] in (1, 3, 4):
                arr = v.permute(0, 2, 3, 1).numpy()  # (T,C,H,W) -> (T,H,W,C)
            elif v.shape[0] in (1, 3, 4):
                arr = v.permute(1, 2, 3, 0).numpy()  # (C,T,H,W) -> (T,H,W,C)
            else:
                raise ValueError(f"Cannot infer tensor video layout from shape {tuple(v.shape)}")
            return [_frame_to_pil(frame) for frame in arr]

        raise TypeError(f"Unsupported video type: {type(video)}")

    def process(
        self,
        interactions: Union[str, List[str]],
        images: Optional[Image.Image] = None,
        videos: Optional[List[Image.Image]] = None,
        size: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        
        self.operators.get_interaction(interactions)
        operator_condition = self.operators.process_interaction()
        self.operators.delete_last_interaction()
        
        visual_context = self.operators.process_perception(images=images, videos=videos, size=size)

        return {
            "operator_condition": operator_condition,
            "visual_context": visual_context
        }

    def __call__(
        self,
        prompt: Optional[str] = "",
        interactions: Optional[Union[str, List[str]]] = None,
        interaction_speeds: Optional[Union[float, List[float]]] = None,
        interaction_distances: Optional[Union[float, List[float]]] = None,
        images: Optional[Image.Image] = None,
        videos: Optional[List[Image.Image]] = None,
        size: Optional[str] = "704*1280",
        seed: Optional[int] = None,
        task_type: Optional[str] = "i2v",
        num_euler_timesteps: Optional[int] = 100,
    ) -> torch.Tensor:

        output_dict = self.process(
            interactions=interactions,
            images=images,
            videos=videos,
            size=size
        )

        output_video = self.synthesis_model.predict(
            prompt=prompt,
            image=output_dict["visual_context"]["ref_images"],
            video=output_dict["visual_context"]["ref_videos"], # pixel values of the video frames
            interactions=interactions, # list of interaction strings, e.g., ["forward", "camera_l"]
            interaction_captions=output_dict["operator_condition"], # list of interaction captions, e.g., ["The camera pushes forward (W).", "The camera pans to the left (←)."]
            interaction_speeds=interaction_speeds,
            interaction_distances=interaction_distances,
            task_type=task_type,
            size=size,
            seed=seed,
            num_euler_timesteps=num_euler_timesteps
        )

        return output_video

    def stream(
        self,
        prompt: Optional[str] = "",
        interactions: Optional[Union[str, List[str]]] = None,
        interaction_speeds: Optional[Union[float, List[float]]] = None,
        interaction_distances: Optional[Union[float, List[float]]] = None,
        images: Optional[Image.Image] = None,
        videos: Optional[List[Image.Image]] = None,
        size: Optional[str] = "704*1280",
        seed: Optional[int] = None,
        task_type: Optional[str] = "i2v",
        num_euler_timesteps: Optional[int] = 100,
    ) -> torch.Tensor:
        if self.memory_module is None:
            raise ValueError("memory_module is None")

        rank = self._dist_rank()

        if isinstance(interactions, str):
            interactions = [interactions]
        if interactions is None or len(interactions) == 0:
            raise ValueError("interactions must be provided in stream().")

        if isinstance(interaction_speeds, (float, int)):
            interaction_speeds = [float(interaction_speeds)] * len(interactions)
        if isinstance(interaction_distances, (float, int)):
            interaction_distances = [float(interaction_distances)] * len(interactions)

        if videos is not None and not (isinstance(videos, list) and (len(videos) == 0 or isinstance(videos[0], Image.Image))):
            videos = self._to_pil_frames(videos)

        if images is not None or videos is not None:
            visual_context = self.operators.process_perception(images=images, videos=videos, size=size)
            input_data = images if images is not None else videos
            self.memory_module.record(
                input_data,
                visual_context=visual_context,
                as_context=True,
                record_frames=False,
            )

        ctx = self.memory_module.select_context()
        if task_type != "t2v" and ctx is None:
            raise ValueError("No context in memory. Provide 'images' or 'videos' in the first stream() call.")

        self.operators.get_interaction(interactions)
        operator_condition = self.operators.process_interaction()
        self.operators.delete_last_interaction()

        effective_task_type = task_type
        predict_image = None if ctx is None else ctx.get("ref_images")
        predict_video = None if ctx is None else ctx.get("ref_videos")

        # In streaming continuation, prefer i2v with the latest frame as anchor.
        if task_type == "i2v" and images is None and videos is None and predict_image is None:
            if predict_video is not None and getattr(self.memory_module, "n_generated_segments", 0) > 0:
                effective_task_type = "v2v"
            else:
                raise ValueError("No valid image/video context for i2v stream continuation.")

        if effective_task_type == "i2v":
            predict_video = None

        output_video = self.synthesis_model.predict(
            prompt=prompt,
            image=predict_image,
            video=predict_video,
            interactions=interactions,
            interaction_captions=operator_condition,
            interaction_speeds=interaction_speeds,
            interaction_distances=interaction_distances,
            task_type=effective_task_type,
            size=size,
            seed=seed,
            num_euler_timesteps=num_euler_timesteps,
        )

        output_video_frames = self._to_pil_frames(output_video)
        if len(output_video_frames) == 0:
            raise RuntimeError("Synthesis returned an empty video in stream().")
        output_visual_context = self.operators.process_perception(
            images=output_video_frames[-1],
            videos=output_video_frames,
            size=size
        )
        self.memory_module.record(
            output_video_frames,
            visual_context=output_visual_context,
            record_frames=(rank == 0),
        )

        return output_video
