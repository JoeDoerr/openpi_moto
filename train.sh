#!/bin/bash

# LoRA fine-tuning for Motoman robot (recommended - memory efficient)
python scripts/train.py motoman_lora --exp_name motoman_test

# To see all available configs, run:
# python scripts/train.py --help