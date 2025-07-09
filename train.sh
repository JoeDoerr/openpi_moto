#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3
export HF_LEROBOT_HOME=/common/home/jhd79/imitation/lerobot_data

echo $HF_LEROBOT_HOME

#Compute normalization statistics
python scripts/compute_norm_stats.py --config-name=motoman_lora

# lora fine-tuning for motoman
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py motoman_lora --exp-name=motoman_test_1 --overwrite

# To see all available configs, run:
# python scripts/train.py --help
