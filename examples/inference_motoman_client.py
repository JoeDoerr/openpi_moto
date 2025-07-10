import dataclasses
import time
import numpy as np
import jax
import zmq
import pickle

from openpi.models import model as _model
from openpi.policies import moto_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

context = zmq.Context()
socket = context.socket(zmq.REQ) #REQ sends requests and is paired with a REP socket that sends replies
socket.connect("tcp://0.0.0.0:8091")

# Run inference on a dummy example. This example corresponds to observations produced by the DROID runtime.
while True:
    example = moto_policy.make_motoman_example() #Will repack then the data transforms will be applied
    socket.send_multipart([b"example", pickle.dumps(example)])
    result = socket.recv_multipart()
    print("Result:", result, type(result))
    time.sleep(1)