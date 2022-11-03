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




if __name__ == "__main__":
    # ENV setup
    env = simple_world_comm_v2.env()
    env.reset()
    