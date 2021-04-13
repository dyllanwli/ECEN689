import random
from collections import namedtuple
import torch
import numpy as np

Experience = namedtuple('Experience',('state', 'action', 'reward', 'new_state', 'done'))