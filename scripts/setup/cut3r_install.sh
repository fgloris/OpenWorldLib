#!/bin/bash
# scripts/setup/cut3r_install.sh
# Description: Setup environment for cut3r-based experiments in SceneFlow
# Usage: bash scripts/setup/cut3r_install.sh

echo "=== [1/4] Installing core numerical and DL stack ==="
pip install "numpy==1.26.4"
pip install torch torchvision

echo "=== [2/4] Installing vision and logging dependencies ==="
pip install roma opencv-python matplotlib pillow==10.3.0 tensorboard tqdm

echo "=== [3/4] Installing geometry and rendering dependencies ==="
pip install trimesh "pyglet<2" viser lpips

echo "=== [4/4] Installing NLP / HF and utility dependencies ==="
pip install "huggingface-hub[torch]>=0.22" gradio scipy einops hydra-core h5py accelerate transformers scikit-learn simple_knn

echo "=== cut3r environment setup completed! ==="
