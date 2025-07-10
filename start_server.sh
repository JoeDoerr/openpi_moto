#!/bin/bash

#python examples/inference_motoman.py
python examples/inference_motoman_server.py --port=8091 --config_name=motoman_lora --checkpoint_dir=./checkpoints/motoman_lora/motoman_test_1/249
