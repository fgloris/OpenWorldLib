import sys
from openworldlib.pipelines.thinksound.pipeline_thinksound import ThinkSoundPipeline
import torchaudio
from pathlib import Path
from loguru import logger


def save_audio_result(result):
    audio = result["audio"]  
    sampling_rate = result["sampling_rate"]
    waveform = audio[0]

    save_path = f"./thinksound_testoutput.wav"
    torchaudio.save(str(save_path), waveform, sampling_rate)



# thinksound不允许为none，duration-sec必须是匹配的
video_path = "./data/test_video_case1/talking_man.mp4"
title = "play guitar"
description = "A man is playing guitar gently"
model_path = "FunAudioLLM/ThinkSound"

pipeline = ThinkSoundPipeline.from_pretrained(
    model_path=model_path,
    model_config="src/openworldlib/synthesis/audio_generation/thinksound/ThinkSound/ThinkSound/configs/model_configs/thinksound.json",
    duration_sec=3.0,
    seed=42,
    compile=False,
    video_dir="videos",
    cot_dir="cot_coarse",
    results_dir="results",
    scripts_dir=".",
    device=None,  # 自动检测设备
)

result = pipeline(
    video_path=video_path,
    title=title,
    description=description,
    use_half=False,
    cfg_scale=5.0,
    num_steps=24,
)

save_audio_result(result)

