import os
import json
import cv2

from PIL import Image

from openworldlib.pipelines.matrix_game.pipeline_matrix_game_3 import MatrixGame3Pipeline

def load_video_to_pil(video_path):
    """读取视频文件并转换为 PIL Image 列表"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found at {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR 转换为 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    print(f"Loaded {len(frames)} frames from {video_path}")
    return frames

def main():
    # Keep test style close to MatrixGame2: a default pretrained model id.
    pretrained_model_path = "Skywork/Matrix-Game-3.0"

    images = load_video_to_pil("./data/test_case/test_video_case1/origin-1.mp4")[:240]

    pipeline = MatrixGame3Pipeline.from_pretrained(model_path=pretrained_model_path, device="cuda")

    case = {
            "name": "nav_left_right",
            "prompt": "A man is talking fast to a women in a kitchen",
            "interactions": ["nomove", "nomove", "nomove"],
            "save_name": "matrix_game_3_demo",
        }

    result = pipeline.v2v(
        images=images,
        prompt=case["prompt"],
        interactions=case["interactions"],
        output_dir="./output",
        save_name=case["save_name"],
        size="704*1280",
        num_iterations=8,
        num_inference_steps=3,
        fa_version="0",
        save_video=False,
        return_result=True,
    )
    video_tensor = result.get("video_tensor")
    if video_tensor is not None:
        saved_path = pipeline.save_video_tensor(video_tensor, "./matrix_game_3_demo.mp4")
    else:
        saved_path = None
    case["video_path"] = saved_path
    case["video_tensor_shape"] = list(video_tensor.shape) if video_tensor is not None else None
    case["custom_video_exists"] = bool(saved_path and os.path.exists(saved_path))

if __name__ == "__main__":
    main()
