import numpy as np
import torch
from gym.spaces import Dict
from gym.spaces.utils import flatten_space
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

class TFJobSearchModelV0(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "observation" in orig_space.spaces
        )
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        
        self.internal_model = FullyConnectedNetwork(
            flatten_space(orig_space["observation"]),
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )
        
        # disable action masking --> will likely lead to invalid actions
        self.no_masking = model_config["custom_model_config"].get("no_masking", False)
        
    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]
        
        # Compute the unmasked logits using only the observation columns of the flattened observation.
        logits, _ = self.internal_model({"obs": input_dict["obs_flat"][:,self.num_outputs:]})        
        
        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()

class JobSearchModelV0(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "observation" in orig_space.spaces
        )        
        
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs)
        nn.Module.__init__(self)
        
        self.internal_model = TorchFC(
            flatten_space(orig_space["observation"]),
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )
        # disable action masking --> will likely lead to invalid actions
        self.no_masking = False
        if "no_masking" in model_config["custom_model_config"]:
            self.no_masking = model_config["custom_model_config"]["no_masking"]
        
    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]
        
        # Compute the unmasked logits using only the observation columns of the flattened observation.
        logits, _ = self.internal_model({"obs": input_dict["obs_flat"][:,self.num_outputs:]})        
        
        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask
        
        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()