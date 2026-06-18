#!/bin/bash
#SBATCH --job-name=cnn-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/cnn-train_%j.log

set -eo pipefail

source ~/miniforge3/bin/activate cnn-env
export PYTHONPATH=.:${PYTHONPATH:-}

# Add pip-installed CUDA libs to LD_LIBRARY_PATH using bash glob
# (avoids python subshell inheritance issues in batch mode)
SITE=~/miniforge3/envs/cnn-env/lib/python3.10/site-packages
for d in $SITE/nvidia/*/lib; do
    export LD_LIBRARY_PATH=$d:${LD_LIBRARY_PATH:-}
done
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

echo "=== GPU info ==="
nvidia-smi

echo "=== Verifying GPU visible to TF ==="
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Synchronous CUDA errors so we get the real message instead of a generic abort
export CUDA_LAUNCH_BLOCKING=1

echo "=== Training ResNet-34 ==="
python train.py --config configs/resnet34.yaml --save saved_models/resnet34.keras

echo "=== Training Xception ==="
python train.py --config configs/xception.yaml --save saved_models/xception.keras

echo "=== Evaluating ResNet-34 ==="
python evaluate.py --model saved_models/resnet34.keras

echo "=== Evaluating Xception ==="
python evaluate.py --model saved_models/xception.keras

echo "=== Done ==="
