import os
import cv2
import imageio
import numpy as np
from PIL import Image
from diffusers.utils import export_to_video

# 假设 pipeline 已经在 openworldlib 中更新了 v2v 方法
from openworldlib.pipelines.infinite_world.pipeline_infinite_world import InfiniteWorldPipeline

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

def save_uint8_video(video_frames, output_path, fps=30):
    """保存视频，处理 uint8 溢出问题"""
    with imageio.get_writer(output_path, fps=fps, quality=8) as writer:
        for frame in video_frames:
            frame = np.asarray(frame)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            writer.append_data(frame)

# 1. 路径设置
# 假设你现在要读取一个现有的视频文件进行续写
video_input_path = "./data/test_case/test_video_case1/talking_man.mp4" 
pretrained_model_path = os.environ.get("INFINITE_WORLD_MODEL_PATH", "MeiGen-AI/Infinite-World")

# 2. 初始化 Pipeline
pipeline = InfiniteWorldPipeline.from_pretrained(
    model_path=pretrained_model_path,
    device="cuda",
)

# 3. 读取输入视频
input_frames = load_video_to_pil(video_input_path)

# 4. 调用 v2v 接口进行续写
# 注意：这里传入的是 video_frames 列表，而不是单张 images
output_video = pipeline.v2v_2(
    video_frames=input_frames,
    prompt="A serene campus walkway lined with modern glass buildings and soft daylight.",
    interactions=["forward", "forward+camera_r", "forward", "camera_l"],
    num_frames=80,      # 续写的帧数
    size=(384, 1024),   # 确保分辨率与模型要求一致
)

# 5. 保存结果
output_path = "infinite_world_v2v_demo.mp4"
save_uint8_video(output_video, output_path, fps=30)
print(f"Success! Extended video saved to {output_path}")