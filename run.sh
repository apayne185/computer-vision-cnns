#!/bin/bash
#SBATCH --job-name=cnn-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/cnn-train_%j.log

set -eo pipefail

source ~/miniforge3/bin/activate cnn-env
export PYTHONPATH=.:${PYTHONPATH:-}

# Make pip-installed CUDA libraries visible to TensorFlow
export LD_LIBRARY_PATH=$(python -c "
import os, glob, site
paths = glob.glob(os.path.join(site.getsitepackages()[0], 'nvidia', '*', 'lib'))
print(':'.join(paths))
"):/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

echo "=== GPU info ==="
nvidia-smi

echo "=== Training ResNet-34 ==="
python train.py --config configs/resnet34.yaml --save saved_models/resnet34.keras

echo "=== Training Xception ==="
python train.py --config configs/xception.yaml --save saved_models/xception.keras

echo "=== Evaluating ResNet-34 ==="
python evaluate.py --model saved_models/resnet34.keras

echo "=== Evaluating Xception ==="
python evaluate.py --model saved_models/xception.keras

echo "=== Done ==="
