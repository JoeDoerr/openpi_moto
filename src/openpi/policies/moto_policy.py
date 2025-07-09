"""
Motoman dual arm robot policy transforms for π₀ compatibility.

This module provides simple transforms for Motoman dual arm robots (16 DOF),
similar to the droid policy approach.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

motoman_dof = 16

def make_motoman_example() -> dict:
    """Creates a random input example for your robot policy."""
    return {
        "state": np.random.rand(motoman_dof),  # Motoman dual arm: 16 DOF with gripper
        "image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),  # Optional
        "prompt": "perform manipulation task",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to π₀ format (uint8 HWC)."""
    image = np.asarray(image)
    # LeRobot stores as float32, convert to uint8
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    # LeRobot stores as CHW, convert to HWC
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

@dataclasses.dataclass(frozen=True)
class MotoPolicyInputs(transforms.DataTransformFn):
    """
    Motoman dual arm robot input transforms for π₀ compatibility.
    """
    
    # The action dimension of the π₀ model (e.g., 32 for base models)
    action_dim: int
    
    # Which π₀ model variant to use  
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        #print("CALLING MOTO POLICY TO TRANSFORM INPUTS", flush=True)
        # Simple state handling - just use the state as-is and pad
        state = np.asarray(data["state"])
        state = transforms.pad_to_dim(state, self.action_dim)
        
        # Parse images to π₀ format (uint8 HWC)
        base_image = _parse_image(data["image"])
        
        # Handle optional wrist image
        if "wrist_image" in data:
            wrist_image = _parse_image(data["wrist_image"])
        else:
            wrist_image = np.zeros_like(base_image)

        match self.model_type:
            case _model.ModelType.PI0:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        
        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }
        
        # Handle actions (training only)
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = transforms.pad_to_dim(actions, self.action_dim)
            inputs["actions"] = actions
        
        # Handle language prompts
        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]
            
        return inputs


@dataclasses.dataclass(frozen=True)
class MotoPolicyOutputs(transforms.DataTransformFn):
    """
    This class converts outputs from the model back to Motoman's native format.
    Used for inference only.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first motoman_dof dims (Motoman's DOF).
        return {"actions": np.asarray(data["actions"][:, :motoman_dof])} 