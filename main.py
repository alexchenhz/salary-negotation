# From https://towardsdatascience.com/using-pettingzoo-with-rllib-for-multi-agent-deep-reinforcement-learning-5ff47c677abd

from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo.butterfly import pistonball_v5
import supersuit as ss
import torch
from torch import nn


# class Agent(nn.Module):
#     pass

class CNNModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
        nn.Conv2d( 3, 32, [8, 8], stride=(4, 4)),
        nn.ReLU(),
        nn.Conv2d( 32, 64, [4, 4], stride=(2, 2)),
        nn.ReLU(),
        nn.Conv2d( 64, 64, [3, 3], stride=(1, 1)),
        nn.ReLU(),
        nn.Flatten(),
        (nn.Linear(3136,512)),
        nn.ReLU(),
        )
        self.policy_fn = nn.Linear(512, num_outputs)
        self.value_fn = nn.Linear(512, 1)
    
    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict[“obs”].permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state
    
    def value_function(self):
        return self._value_out.flatten()


if __name__ == "__main__":
    # ENV setup
    env = simple_world_comm_v2.env()
    env.reset()
    