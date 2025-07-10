import dataclasses
import time
import numpy as np
import jax
import tyro

from openpi.models import model as _model
from openpi.policies import moto_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import zmq
import json
import pickle


@dataclasses.dataclass
class Args:
    """Arguments for the Motoman inference server."""
    
    # ZMQ server configuration
    port: int = 8091
    host: str = "0.0.0.0"
    
    # Model configuration
    config_name: str = "motoman_lora"
    checkpoint_dir: str = "./checkpoints/motoman_lora/motoman_test_1/249"
    
    # Optional: whether to download from GCS if local path doesn't exist
    download_if_missing: bool = False


def main(args: Args):
    context = zmq.Context()
    socket = context.socket(zmq.REP) #REP sends replies when it gets something and is paired with a REQ socket that sends requests
    socket.bind(f"tcp://{args.host}:{args.port}")
    print(f"Server is running on {args.host}:{args.port}")

    config = _config.get_config(args.config_name)
    
    # Optionally download checkpoint if local path doesn't exist and download flag is set
    checkpoint_path = args.checkpoint_dir
    if args.download_if_missing:
        checkpoint_path = download.maybe_download(args.checkpoint_dir)
    
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Create a trained policy.
    policy = _policy_config.create_trained_policy(config, checkpoint_path)

    while True:
        input_data = socket.recv_multipart()
        print("Received input data")
        name = input_data[0].decode()
        if name == "exit":
            break
        model_input = pickle.loads(input_data[1])
        result = policy.infer(model_input)
        print("Result:", result, type(result), name)
        socket.send_multipart([b"result", pickle.dumps(result)])
        #time.sleep(1)

    del policy
    socket.close()
    context.term()


if __name__ == "__main__":
    main(tyro.cli(Args))