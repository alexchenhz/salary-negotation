from collections import OrderedDict
import functools
import random

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, Dict, Tuple, Text

from gymnasium.spaces.utils import flatten, flatdim, unflatten

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

# space = Dict({"test1": Tuple((Discrete(4), Discrete(3))), "test2": Discrete(2)})

# sample = OrderedDict([('test1', (0, 0)), ('test2', 0)])

space = Tuple((Discrete(4), Discrete(2)))

sample = (3,1)

print(space, np.zeros(flatdim(space)))

print(sample)

print(flatten(space, sample))

x = flatten(space, sample)

print(unflatten(space, x))