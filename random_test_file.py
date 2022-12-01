import functools
import random

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, Dict, Tuple, Text

from gymnasium.spaces.utils import flatten, flatdim

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

space = Tuple((Discrete(4), Discrete(3)))

sample = space.sample()

print(space, (0,0), np.zeros(flatdim(space)))

x = np.zeros(7)
x[0] = 1
print(x)