import os
import numpy as np
import torch
from typing import List, Optional, Union, Dict, Any
from PIL import Image
import base64
import io

from .base_operator import BaseOperator
from openworldlib.representations.point_clouds_generation.flash_world.flash_world.utils import (
    matrix_to_quaternion,
)


def _look_at_rotation_wxyz_columns(
    eye: np.ndarray,
    target: np.ndarray,
    world_up: np.ndarray,
) -> np.ndarray:
    """
    Camera-to-world rotation (columns are camera X/Y/Z axes in world frame).
    Matches FlashWorld `create_rays`: viewing direction in world is -R[:,2]
    (camera looks down -Z), so we set R[:,2] = -forward where forward points
    from camera toward the look target.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    world_up = np.asarray(world_up, dtype=np.float64)
    forward = target - eye
    n = float(np.linalg.norm(forward))
    if n < 1e-8:
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        forward = forward / n
    right = np.cross(world_up, forward)
    rn = float(np.linalg.norm(right))
    if rn < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right = right / rn
    up = np.cross(forward, right)
    # Columns: right, up, -forward so that R @ [0,0,-1] = forward
    return np.stack([right, up, -forward], axis=1)


def _rotation_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64
    )


def _rotation_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _pose_to_quaternion_wxyz(
    position_xyz,
    yaw_world: float = 0.0,
    pitch_cam: float = 0.0,
) -> list:
    """
    Look at origin, optional pitch in camera space (camera_up / camera_down),
    then optional yaw around world +Y (camera_l / camera_r).
    Rotation order for column vectors: R_total = R_y(yaw) @ R_look @ R_x(pitch).
    """
    eye = np.asarray(position_xyz, dtype=np.float64)
    R_look = _look_at_rotation_wxyz_columns(
        eye, np.zeros(3), np.array([0.0, 1.0, 0.0])
    )
    R = R_look
    if abs(pitch_cam) > 1e-8:
        R = R @ _rotation_x(pitch_cam)
    if abs(yaw_world) > 1e-8:
        R = _rotation_y(yaw_world) @ R
    q = matrix_to_quaternion(torch.from_numpy(R).float().unsqueeze(0))[0]
    return [float(q[i]) for i in range(4)]


def _camera_forward_right_world(eye: np.ndarray):
    """View toward origin: forward_world points from camera into the scene."""
    eye = np.asarray(eye, dtype=np.float64)
    forward = -eye
    fn = float(np.linalg.norm(forward))
    if fn < 1e-8:
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        forward = forward / fn
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(world_up, forward)
    rn = float(np.linalg.norm(right))
    if rn < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right = right / rn
    return forward, right


class FlashWorldOperator(BaseOperator):
    """Operator for FlashWorld pipeline utilities."""
    
    def __init__(
        self,
        operation_types=["textual_instruction", "action_instruction", "visual_instruction"],
        interaction_template=[
            "text_prompt",
            "forward", "backward", "left", "right",
            "camera_up", "camera_down", "camera_l", "camera_r",
            "camera_zoom_in", "camera_zoom_out"
        ]
    ):
        """
        Initialize FlashWorld operator.
        
        Args:
            operation_types: List of operation types
            interaction_template: List of valid interaction types
                - "text_prompt": Text description for scene generation
                - "forward/backward/left/right": Dolly/strafe along view / camera right (after look-at)
                - "camera_up/camera_down": Tilt (pitch) in camera space
                - "camera_l/camera_r": Pan (yaw around world +Y)
                - "camera_zoom_in/out": Focal length ramp over the clip (repeat to strengthen)
        """
        super(FlashWorldOperator, self).__init__(operation_types=operation_types)
        self.interaction_template = interaction_template
        self.interaction_template_init()
    
    def check_interaction(self, interaction):
        """
        Check if interaction is in the interaction template.
        
        Args:
            interaction: Interaction string to check
            
        Returns:
            True if interaction is valid
            
        Raises:
            ValueError: If interaction is not in template
        """
        if interaction not in self.interaction_template:
            raise ValueError(f"{interaction} not in template. Available: {self.interaction_template}")
        return True
    
    def get_interaction(self, interaction):
        """
        Add interaction to current_interaction list after validation.
        
        Args:
            interaction: Interaction string to add
        """
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)
    
    def process_interaction(
        self, 
        num_frames: Optional[int] = None,
        image_width: int = 704,
        image_height: int = 480
    ) -> Dict[str, Any]:
        """
        Process current interactions and convert to features for representation/synthesis.
        Converts camera actions to actual camera parameters that can be used by representation.
        
        Args:
            num_frames: Number of frames for video generation (optional)
            image_width: Image width for camera intrinsics
            image_height: Image height for camera intrinsics
            
        Returns:
            Dictionary containing processed interaction features:
                - text_prompt: str, text description (if provided)
                - cameras: List[Dict], camera parameters for each frame
                - num_frames: int, number of frames
        """
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process. Use get_interaction() first.")
        
        # Get the latest interaction
        latest_interaction = self.current_interaction[-1]
        self.interaction_history.append(latest_interaction)
        
        num_frames = num_frames or 16
        
        # Extract text prompts and camera / movement actions (both matter for trajectory)
        text_prompt = ""
        camera_actions = []
        for interaction in self.current_interaction:
            if interaction == "text_prompt":
                # Text prompt should be passed separately via pipeline `prompt`
                continue
            # Include pan/tilt/zoom (camera_*) and dolly/strafe (forward, backward, left, right, ...)
            camera_actions.append(interaction)
        
        # Convert camera actions to camera parameters
        cameras = self._camera_actions_to_cameras(
            camera_actions=camera_actions,
            num_frames=num_frames,
            image_width=image_width,
            image_height=image_height
        )
        
        result = {
            "text_prompt": text_prompt,
            "cameras": cameras,
            "num_frames": num_frames,
        }
        
        return result
    
    def _camera_actions_to_cameras(
        self,
        camera_actions: List[str],
        num_frames: int,
        image_width: int,
        image_height: int
    ) -> List[Dict[str, Any]]:
        """
        Convert camera action strings to camera parameter dictionaries.

        Each token in ``interaction_template`` (except ``text_prompt``) is counted;
        repeated tokens strengthen that effect. Base path: circular orbit around
        origin with radius ``radius`` and center offset ``base_position``.

        - ``forward`` / ``backward``: move along view toward / away from the look
          target (camera-relative dolly, scaled by time ``t``).
        - ``left`` / ``right``: strafe along camera right (horizontal, in-world).
        - ``camera_up`` / ``camera_down``: tilt (pitch) in camera space.
        - ``camera_l`` / ``camera_r``: pan (yaw around world +Y); both cancel out.
        - ``camera_zoom_in`` / ``camera_zoom_out``: ramp ``fx``/``fy`` over the clip.

        Args:
            camera_actions: List of camera action strings
            num_frames: Number of frames
            image_width: Image width
            image_height: Image height

        Returns:
            List of camera dictionaries with position, quaternion, and intrinsics
        """
        if not camera_actions:
            # Default circular camera path
            return self._create_default_cameras(num_frames, image_width, image_height)
        
        # Count each instruction so stacked interactions accumulate strength
        n_forward = sum(1 for a in camera_actions if a == "forward")
        n_backward = sum(1 for a in camera_actions if a == "backward")
        n_left = sum(1 for a in camera_actions if a == "left")
        n_right = sum(1 for a in camera_actions if a == "right")
        n_cam_up = sum(1 for a in camera_actions if a == "camera_up")
        n_cam_down = sum(1 for a in camera_actions if a == "camera_down")
        n_zoom_in = sum(1 for a in camera_actions if a == "camera_zoom_in")
        n_zoom_out = sum(1 for a in camera_actions if a == "camera_zoom_out")

        has_camera_l = "camera_l" in camera_actions
        has_camera_r = "camera_r" in camera_actions
        yaw_sign = float(has_camera_r) - float(has_camera_l)

        yaw_max = np.pi / 2
        pitch_max = np.pi / 4
        dolly_scale = 0.45
        zoom_in_strength = 0.22
        zoom_out_strength = 0.18

        cameras = []
        radius = 2.0
        base_position = np.array([0.0, 0.5, 2.0], dtype=np.float64)
        denom = max(num_frames - 1, 1)
        nf = max(num_frames, 1)

        for i in range(num_frames):
            angle = 2 * np.pi * i / nf
            t = i / denom

            x = radius * np.cos(angle) + base_position[0]
            z = radius * np.sin(angle) + base_position[2]
            y = float(base_position[1])
            eye = np.array([x, y, z], dtype=np.float64)

            # Dolly / strafe in camera frame (toward scene = forward, horizontal = right)
            fwd_w, right_w = _camera_forward_right_world(eye)
            dolly = (n_forward - n_backward) * dolly_scale * t
            strafe = (n_right - n_left) * dolly_scale * t
            eye = eye + fwd_w * dolly + right_w * strafe

            # Tilt: camera_up / camera_down (pitch in camera space)
            pitch_cam = float(n_cam_up - n_cam_down) * pitch_max * t

            # Pan: camera_l / camera_r (yaw around world +Y)
            yaw_world = yaw_sign * yaw_max * t

            quat = _pose_to_quaternion_wxyz(
                eye, yaw_world=yaw_world, pitch_cam=pitch_cam
            )

            # Zoom: focal length ramps over the clip; repeat tokens strengthen the effect
            zoom_factor = 1.0 + n_zoom_in * zoom_in_strength * t - n_zoom_out * zoom_out_strength * t
            zoom_factor = float(np.clip(zoom_factor, 0.55, 2.0))

            camera = {
                "position": [float(eye[0]), float(eye[1]), float(eye[2])],
                "quaternion": quat,
                "fx": image_width * 0.7 * zoom_factor,
                "fy": image_height * 0.7 * zoom_factor,
                "cx": image_width * 0.5,
                "cy": image_height * 0.5,
            }
            cameras.append(camera)

        return cameras
    
    def _create_default_cameras(
        self,
        num_frames: int,
        image_width: int,
        image_height: int
    ) -> List[Dict[str, Any]]:
        """
        Create default camera trajectory (circular path).
        
        Args:
            num_frames: Number of frames
            image_width: Image width
            image_height: Image height
            
        Returns:
            List of camera dictionaries
        """
        cameras = []
        radius = 2.0
        
        for i in range(num_frames):
            angle = 2 * np.pi * i / num_frames

            # Circular camera path
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            y = 0.5

            quat = _pose_to_quaternion_wxyz([x, y, z], yaw_world=0.0, pitch_cam=0.0)
            
            camera = {
                'position': [float(x), float(y), float(z)],
                'quaternion': quat,
                'fx': image_width * 0.7,
                'fy': image_height * 0.7,
                'cx': image_width * 0.5,
                'cy': image_height * 0.5,
            }
            cameras.append(camera)
        
        return cameras
    
    def process_perception(
        self,
        input_signal: Union[str, np.ndarray, torch.Tensor, Image.Image, bytes]
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Process visual signal (image) for real-time interactive updates.
        
        Args:
            input_signal: Visual input signal - can be:
                - Image file path (str)
                - Numpy array (H, W, 3) in RGB format
                - Torch tensor (C, H, W) or (1, C, H, W) in CHW format
                - PIL Image
                - Base64 encoded image string
                - Bytes of image data
                
        Returns:
            PIL Image in RGB format
            
        Raises:
            ValueError: If image cannot be loaded or processed
        """
        if isinstance(input_signal, Image.Image):
            # Already a PIL Image, convert to RGB
            return input_signal.convert('RGB')
        
        elif isinstance(input_signal, str):
            # Check if it's a file path or base64
            if os.path.exists(input_signal):
                # File path
                image = Image.open(input_signal)
                return image.convert('RGB')
            elif input_signal.startswith('data:image'):
                # Base64 encoded image
                if ',' in input_signal:
                    image_data = input_signal.split(',', 1)[1]
                else:
                    image_data = input_signal
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                return image.convert('RGB')
            else:
                raise ValueError(f"Invalid input: {input_signal}")
        
        elif isinstance(input_signal, bytes):
            # Bytes data
            image = Image.open(io.BytesIO(input_signal))
            return image.convert('RGB')
        
        elif isinstance(input_signal, np.ndarray):
            # Numpy array
            if input_signal.max() <= 1.0:
                input_signal = (input_signal * 255).astype(np.uint8)
            else:
                input_signal = input_signal.astype(np.uint8)
            
            # Convert BGR to RGB if needed
            if len(input_signal.shape) == 3 and input_signal.shape[2] == 3:
                if input_signal[..., 0].mean() > input_signal[..., 2].mean():
                    input_signal = input_signal[..., ::-1]
            
            image = Image.fromarray(input_signal)
            return image.convert('RGB')
        
        elif isinstance(input_signal, torch.Tensor):
            # Torch tensor
            if input_signal.dim() == 3:
                image_array = input_signal.permute(1, 2, 0).cpu().numpy()
            else:
                image_array = input_signal[0].permute(1, 2, 0).cpu().numpy()
            
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)
            
            image = Image.fromarray(image_array)
            return image.convert('RGB')
        
        else:
            raise ValueError(f"Unsupported input type: {type(input_signal)}")
    
    def delete_last_interaction(self):
        """Delete the last interaction from current_interaction list."""
        if len(self.current_interaction) > 0:
            self.current_interaction = self.current_interaction[:-1]
        else:
            raise ValueError("No interaction to delete.")

