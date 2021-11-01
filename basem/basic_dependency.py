import numpy as np
import os
import pandas as pd
import math
import random


"""
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


from scipy.signal import butter, lfilter
from scipy import signal
from scipy.stats import mode
import scipy.io
"""


import torch
from torch import nn

import torch.nn.functional as F
from torch.nn import Linear
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from torch.nn.parameter import Parameter
from torch.nn.functional import linear


from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

#import torch.nn.utils.prune as prune
#import torchaudio

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def seed_e(seed_value):
  seed_everything(seed_value)
  random.seed(seed_value)
  np.random.seed(seed_value) 
  torch.manual_seed(seed_value)
  torch.cuda.manual_seed(seed_value)
  torch.cuda.manual_seed_all(seed_value)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

# install: axial_positional_embedding
