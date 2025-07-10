import dataclasses
import time
import numpy as np
import jax

from openpi.models import model as _model
from openpi.policies import moto_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

config = _config.get_config("motoman_lora")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/motoman_lora")

# Create a trained policy.
policy = _policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example. This example corresponds to observations produced by the DROID runtime.
for i in range(10):
    example = moto_policy.make_motoman_example() #Will repack then the data transforms will be applied
    result = policy.infer(example)
    print("Result:", result, type(result))
    time.sleep(1)

print(result["actions"].shape)

# Delete the policy to free up memory.
del policy