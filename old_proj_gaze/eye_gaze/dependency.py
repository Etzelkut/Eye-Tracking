import numpy as np
import torch
from torch import nn
import os
import pandas as pd
import math
import random

from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from torch.nn.parameter import Parameter
from torch.nn.functional import linear
import torch.nn.utils.prune as prune

import torchaudio

from scipy.signal import butter, lfilter
from scipy import signal
from scipy.stats import mode
import scipy.io

import pytorch_lightning as pl
from pytorch_lightning import seed_everything


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


# from onmt.utils.misc import generate_relative_positions_matrix
def generate_relative_positions_matrix(length, max_relative_positions,
                                       cache=False):
    """Generate the clipped relative positions matrix
       for a given length and maximum relative positions"""
    if cache:
        distance_mat = torch.arange(-length+1, 1, 1).unsqueeze(0)
    else:
        range_vec = torch.arange(length)
        range_mat = range_vec.unsqueeze(-1).expand(-1, length).transpose(0, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
    distance_mat_clipped = torch.clamp(distance_mat,
                                       min=-max_relative_positions,
                                       max=max_relative_positions)
    # Shift values to be >= 0
    final_mat = distance_mat_clipped + max_relative_positions
    return final_mat

# dropout tracker: main_block_dict dropout,  ff_dropout,   classificator_dropout  ;;;     not used: dropout_block, dropout_pos_emb
# dropout tracker old: main_block_dict dropout, add_block_dict dropout, classificator_dropout  ;;;     not used: dropout_pos_emb
