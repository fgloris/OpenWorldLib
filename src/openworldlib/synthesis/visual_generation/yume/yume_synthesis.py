import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from diffusers.video_processor import VideoProcessor
from PIL import Image

from ...base_synthesis import BaseSynthesis
from ....base_models.diffusion_model.video.wan_2p1.configs import WAN_CONFIGS
from .yume import YumeI2V, YUME_SIZE_CONFIGS, YUME_MAX_AREA_CONFIGS


def get_sampling_sigmas(sampling_steps, shift):
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    return sigma


class YumeSynthesis(BaseSynthesis):

    def __init__(
        self,
        model,
        device,
        weight_dtype,
    ) -> None:
        super().__init__()
        self.model = model
        self.weight_dtype = weight_dtype
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        device,
        weight_dtype,
        fsdp,
    ) -> "YumeSynthesis":
        torch.backends.cuda.matmul.allow_tf32 = True

        if os.path.isdir(pretrained_model_path):
            model_root = pretrained_model_path
        else:
            from huggingface_hub import snapshot_download

            print(f"Downloading weights from HuggingFace repo: {pretrained_model_path}")
            model_root = snapshot_download(pretrained_model_path)
            print(f"Model downloaded to: {model_root}")

        rank = int(os.environ.get("LOCAL_RANK", "0"))

        import torch.distributed as dist

        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
            device = torch.device(rank)
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)

        cfg = WAN_CONFIGS["i2v-14B"]
        model = YumeI2V(
            config=cfg,
            checkpoint_dir=model_root,
            device_id=rank,
            dit_fsdp=fsdp,
        )
        model.init_model(
            config=cfg,
            checkpoint_dir=model_root,
            device_id=rank,
            t5_cpu=False,
        )

        model.model.eval().requires_grad_(False).to(weight_dtype)
        if not fsdp:
            model.model.to(device)

        return cls(
            model=model,
            device=device,
            weight_dtype=weight_dtype,
        )

    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        frame = tensor.detach().float().cpu()
        if frame.ndim != 3:
            raise ValueError(f"Expected frame tensor shape (C,H,W), got {tuple(frame.shape)}")
        if frame.shape[0] == 1:
            frame = frame.repeat(3, 1, 1)
        if frame.shape[0] != 3:
            raise ValueError(f"Expected 3 channels, got {frame.shape[0]}")
        arr = frame.add(1.0).div(2.0).clamp(0, 1).permute(1, 2, 0).numpy()
        arr = (arr * 255.0).astype(np.uint8)
        return Image.fromarray(arr)

    def _build_initial_visual_condition(
        self,
        image: Optional[torch.Tensor],
        video: Optional[torch.Tensor],
        task_type: str,
        size: Tuple[int, int],
        current_frame_num: int,
    ) -> Tuple[torch.Tensor, Image.Image]:
        if task_type == "i2v":
            if image is None:
                if video is None:
                    raise ValueError("i2v requires `image` or `video`.")
                if video.ndim != 4:
                    raise ValueError(f"Expected video tensor shape (F,C,H,W), got {tuple(video.shape)}")
                image = video[0].add(1.0).div(2.0).clamp(0, 1)
            base_video = torch.zeros(
                image.shape[0], 1 + current_frame_num, size[0], size[1], device=self.device
            )
            base_video[:, 0] = (image.to(self.device) - 0.5) * 2
        elif task_type == "v2v":
            if video is None:
                raise ValueError("v2v requires `video`.")
            if video.ndim != 4:
                raise ValueError(f"Expected video tensor shape (F,C,H,W), got {tuple(video.shape)}")
            v = video.to(self.device)
            expected = 1 + current_frame_num
            if v.shape[0] < expected:
                pad = torch.zeros(
                    expected - v.shape[0], v.shape[1], v.shape[2], v.shape[3], device=self.device
                )
                v = torch.cat([v, pad], dim=0)
            v = v[:expected]
            base_video = v.permute(1, 0, 2, 3).contiguous()
        elif task_type == "t2v":
            base_video = torch.zeros(3, 1 + current_frame_num, size[0], size[1], device=self.device)
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        model_input_de = torch.cat(
            [base_video[:, 0].unsqueeze(1).repeat(1, 16, 1, 1), base_video[:, : 1 + current_frame_num]],
            dim=1,
        )
        anchor_img = self._tensor_to_pil(model_input_de[:, 0])
        return model_input_de, anchor_img

    @staticmethod
    def _normalize_sampling_method(sampling_method: Optional[str]) -> str:
        if sampling_method is None:
            return "ode"
        method = sampling_method.lower()
        if method not in ("ode", "sde"):
            raise ValueError(
                f"Unsupported sampling_method: {sampling_method}. Supported values: ['ode', 'sde']"
            )
        return method

    @torch.no_grad()
    def predict_per_interaction(
        self,
        prompt,
        interaction_idx,
        interaction,
        interaction_caption,
        interaction_speed,
        interaction_distance,
        next_interaction,
        next_interaction_caption,
        next_interaction_speed,
        next_interaction_distance,
        img_pil,
        model_input_de,
        task_type,
        size,
        seed,
        max_area,
        num_euler_timesteps,
        sampling_method,
        current_latent_num,
        current_frame_num,
        history_state=None,
    ):
        caption = interaction_caption
        if "camera_" not in interaction.lower():
            caption += f"Actual distance moved: {interaction_distance} at {interaction_speed} meters per second."
        else:
            caption += f"View rotation speed: {interaction_speed}."

        rand_num_img = 0.6
        sample_step_num = num_euler_timesteps
        latent_frame_zero = current_latent_num
        guide_scale = 5.0
        sampling_sigmas = get_sampling_sigmas(sample_step_num, 3.0)

        if interaction_idx == 0:
            prompt = prompt if prompt else ""
            caption = prompt + caption

            _, _, arg_c, noise, model_input, clip_context, arg_null = self.model.generate(
                model_input=model_input_de,
                device=self.device,
                input_prompt=caption,
                img=img_pil,
                max_area=max_area,
                frame_num=model_input_de.shape[1],
                shift=17,
                sample_solver="unipc",
                sampling_steps=50,
                guide_scale=guide_scale,
                seed=seed,
                rand_num_img=rand_num_img,
                offload_model=False,
                flag_sample=True,
            )
            latent = noise
            model_input_1 = None
        else:
            if history_state is None:
                raise RuntimeError("history_state is required for interaction_idx > 0.")
            arg_c = history_state["arg_c"]
            arg_null = history_state["arg_null"]
            model_input = history_state["model_input"]
            model_input_de = history_state["model_input_de"]
            clip_context = history_state["clip_context"]
            model_input_1 = history_state["model_input_1"]
            noise = torch.randn_like(model_input_1)
            latent = noise.clone()

        with torch.autocast("cuda", dtype=self.weight_dtype):
            for i in range(sample_step_num):
                latent_model_input = [latent]
                sigma_i = float(sampling_sigmas[i])
                timestep = torch.tensor([sigma_i * 1000], device=self.device)

                noise_pred_cond, _ = self.model.model(
                    latent_model_input,
                    t=timestep,
                    rand_num_img=rand_num_img,
                    **arg_c,
                )
                noise_pred_uncond, _ = self.model.model(
                    latent_model_input,
                    t=timestep,
                    rand_num_img=rand_num_img,
                    **arg_null,
                )
                noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

                latent_tail = latent[:, -latent_frame_zero:, :, :]
                noise_pred_tail = noise_pred[:, -latent_frame_zero:, :, :]
                if i + 1 == sample_step_num:
                    dsigma = -sigma_i
                    temp_x0 = latent_tail + dsigma * noise_pred_tail
                else:
                    sigma_i_next = float(sampling_sigmas[i + 1])
                    dsigma = sigma_i_next - sigma_i
                    temp_x0 = latent_tail + dsigma * noise_pred_tail

                if sampling_method == "sde":
                    eta = 0.3
                    pred_original_sample = latent_tail + (0 - sigma_i) * noise_pred_tail
                    delta_t = 0.0 if i + 1 == sample_step_num else max(0.0, sigma_i - sigma_i_next)
                    std_dev_t = eta * math.sqrt(delta_t)
                    score_estimate = -(
                        latent_tail - pred_original_sample * (1 - sigma_i)
                    ) / (sigma_i ** 2 + 1e-8)
                    log_term = -0.5 * (eta**2) * score_estimate
                    prev_sample_mean = temp_x0 + log_term * dsigma
                    temp_x0 = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

                sigma_next = sampling_sigmas[min(sample_step_num - 1, i + 1)]
                if interaction_idx > 0:
                    latent = torch.cat(
                        [
                            noise[:, :-latent_frame_zero, :, :] * sigma_next
                            + (1 - sigma_next) * model_input_1[:, :-latent_frame_zero, :, :],
                            temp_x0,
                        ],
                        dim=1,
                    )
                else:
                    latent = torch.cat(
                        [
                            noise[:, :-latent_frame_zero, :, :] * sigma_next
                            + (1 - sigma_next) * model_input[:, :-latent_frame_zero, :, :],
                            temp_x0,
                        ],
                        dim=1,
                    )

        if interaction_idx > 0:
            model_input = torch.cat([model_input, latent[:, -latent_frame_zero:, :, :]], dim=1)
        else:
            model_input = torch.cat(
                [model_input[:, :-latent_frame_zero, :, :], latent[:, -latent_frame_zero:, :, :]],
                dim=1,
            )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            video_cat = self.model.vae.decode([model_input.to(torch.float32)])[0]
            current_video = video_cat[:, -current_frame_num:]

        if interaction_idx > 0:
            model_input_de = torch.cat([model_input_de, current_video[:, -current_frame_num:, :, :]], dim=1)
        else:
            model_input_de = torch.cat(
                [model_input_de[:, :-current_frame_num, :, :], current_video[:, -current_frame_num:, :, :]],
                dim=1,
            )

        next_state = {
            "arg_c": None,
            "arg_null": None,
            "model_input": model_input,
            "model_input_de": model_input_de,
            "model_input_1": None,
            "clip_context": clip_context,
        }

        if next_interaction_caption is not None:
            next_caption = "First-person perspective." + next_interaction_caption
            if "camera_" not in next_interaction.lower():
                next_caption += (
                    f"Actual distance moved: {next_interaction_distance} "
                    f"at {next_interaction_speed} meters per second."
                )
            else:
                next_caption += f"View rotation speed: {next_interaction_speed}."
            next_caption += prompt if prompt else ""

            (
                _,
                _,
                next_arg_c,
                _,
                model_input_1,
                clip_context,
                next_arg_null,
            ) = self.model.generate_next(
                model_input_de,
                model_input,
                device=self.device,
                input_prompt=next_caption,
                img=img_pil,
                max_area=max_area,
                frame_num=model_input.squeeze().shape[1],
                shift=17,
                sample_solver="unipc",
                sampling_steps=50,
                guide_scale=guide_scale,
                seed=seed,
                rand_num_img=rand_num_img,
                offload_model=False,
                clip_context=clip_context,
                flag_sample=True,
            )
            model_input_1 = torch.cat(
                [
                    model_input_1,
                    torch.zeros(
                        16,
                        latent_frame_zero,
                        model_input_1.shape[2],
                        model_input_1.shape[3],
                        device=self.device,
                    ),
                ],
                dim=1,
            )

            next_state.update(
                {
                    "arg_c": next_arg_c,
                    "arg_null": next_arg_null,
                    "model_input_1": model_input_1,
                    "clip_context": clip_context,
                }
            )

        return current_video, next_state

    @torch.no_grad()
    def predict(
        self,
        prompt,
        image,
        video,
        interactions,
        interaction_captions,
        interaction_speeds,
        interaction_distances,
        task_type,
        size,
        seed,
        num_euler_timesteps,
        sampling_method: Optional[str] = None,
    ):
        if size not in YUME_SIZE_CONFIGS:
            raise ValueError(f"Unsupported size: {size}. Supported sizes: {list(YUME_SIZE_CONFIGS.keys())}")
        sampling_method = self._normalize_sampling_method(sampling_method)

        current_latent_num = 8
        current_frame_num = 32
        output_video_list: List[torch.Tensor] = []

        if image is not None:
            image = image.to(self.device)
        if video is not None:
            video = video.to(self.device)

        model_input_de, img_pil = self._build_initial_visual_condition(
            image=image,
            video=video,
            task_type=task_type,
            size=YUME_SIZE_CONFIGS[size],
            current_frame_num=current_frame_num,
        )

        history_state: Optional[Dict[str, torch.Tensor]] = None
        for interaction_idx, interaction_caption in enumerate(interaction_captions):
            next_caption = (
                interaction_captions[interaction_idx + 1]
                if interaction_idx + 1 < len(interaction_captions)
                else None
            )
            next_interaction = interactions[interaction_idx + 1] if next_caption is not None else None
            next_speed = (
                interaction_speeds[interaction_idx + 1] if next_caption is not None else None
            )
            next_distance = (
                interaction_distances[interaction_idx + 1] if next_caption is not None else None
            )

            output_video_per_interaction, history_state = self.predict_per_interaction(
                prompt=prompt,
                interaction_idx=interaction_idx,
                interaction=interactions[interaction_idx],
                interaction_caption=interaction_caption,
                interaction_speed=interaction_speeds[interaction_idx],
                interaction_distance=interaction_distances[interaction_idx],
                next_interaction=next_interaction,
                next_interaction_caption=next_caption,
                next_interaction_speed=next_speed,
                next_interaction_distance=next_distance,
                img_pil=img_pil,
                model_input_de=model_input_de if interaction_idx == 0 else history_state["model_input_de"],
                task_type=task_type,
                size=YUME_SIZE_CONFIGS[size],
                seed=seed,
                max_area=YUME_MAX_AREA_CONFIGS[size],
                num_euler_timesteps=num_euler_timesteps,
                sampling_method=sampling_method,
                current_latent_num=current_latent_num,
                current_frame_num=current_frame_num,
                history_state=history_state,
            )
            output_video_list.append(output_video_per_interaction)

        vae_spatial_scale_factor = 8
        video_processor = VideoProcessor(vae_scale_factor=vae_spatial_scale_factor)
        output_video = video_processor.postprocess_video(
            torch.cat(output_video_list, dim=1).unsqueeze(0),
            output_type="pil",
        )[0]
        return output_video
