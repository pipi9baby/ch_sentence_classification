import os
import math
import torch
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from tqdm import tqdm_notebook as tqdm
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange
import json
from torch.nn import CrossEntropyLoss
import sys
import numpy as np

import visualization as vl
from transformers import BertTokenizer, BertModel
from transformers import XLNetTokenizer, XLNetModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import AdamW, get_linear_schedule_with_warmup

# activate function
class gelu(nn.Module):
    
    def __init__(self):
        super(gelu, self).__init__()

    def forward(self, x):
        cdf = 0.5 * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        return x * cdf


class MutiLabelModel(nn.Module):
    
    def __init__(self, encoder, emb_size=1024, out_size=5, ce_size=23, hidden=256): # hidden=256
        super(MutiLabelModel, self).__init__()
        
        self.encoder = encoder
        self.fn_size = emb_size
        
        self.out_fn = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.fn_size, self.fn_size//2),
            gelu(),
            nn.Dropout(0.2),
            nn.Linear(self.fn_size//2, out_size),
        )
        
    def forward(self, inp, seg, cat_emb=None, cls_loc=0): # , inp_title, seg_inp_title, cls_loc=0):

        embs = self.encoder(inp, seg)[0] # [batch, seq, hidden]
        outputs = embs[:, cls_loc, :]
        outputs = self.out_fn(outputs)

        return outputs