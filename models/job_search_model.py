import argparse
import os
import random

import gym
import numpy as np
import ray
import torch
from gym.spaces import Box, Dict, Discrete, Tuple
from gym.spaces.utils import flatdim, flatten, flatten_space
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.preprocessors import (DictFlatteningPreprocessor,
                                            Preprocessor)
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.tune.registry import register_env
from supersuit import pad_action_space_v0, pad_observations_v0
from torch import nn

import environment.job_search_environment as job_search_env

tf1, tf, tfv = try_import_tf()
torch, _ = try_import_torch()

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