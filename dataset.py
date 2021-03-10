# %%
from inspect import Parameter
import os
import sys, re, math, time
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import collections
from collections import OrderedDict
import  matplotlib.pyplot as plt
from torch._C import dtype
from torch.utils.data import DataLoader, Dataset, RandomSampler
import torch
import pandas as pd

## ######################## ##
#
#  Define CHARSET, CHARLEN
#
## ######################## ## 

VOCAB_PROTEIN = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
				"U": 19, "T": 20, "W": 21, 
				"V": 22, "Y": 23, "X": 24, 
				"Z": 25 }

VOCAB_LIGAND_ISO = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, 
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, 
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

def smiles2int(drug):

    return [VOCAB_LIGAND_ISO[s] for s in drug]

def seqs2int(target):

    return [VOCAB_PROTEIN[s] for s in target] 

## ######################## ##
#
#  DATASET Class
#
## ######################## ## 
# works for large dataset
# For Davis, protein: 1200, ligand: 85
# For KIBA, protein: 1000, ligand: 100

class DataPrepared(object):
  def __init__(self, fpath, ligand_len=100, protein_len=1500):
    self.fpath = fpath

    self.ligand_len = ligand_len
    self.protein_len = protein_len

    self.vocab_protein_len = len(VOCAB_PROTEIN)
    self.vocab_ligand_len = len(VOCAB_LIGAND_ISO)

  def read_sets(self, fold, split_type='warm'): 
    fpath = self.fpath
    filename = split_type + '.kfold'
    print(f"Reading fold_{fold} from {fpath}")

    with open(os.path.join(fpath, filename), 'rb') as f:
        kfold = pickle.load(f)

    return kfold[f"fold_{fold}"]

  def get_data(self): 
    fpath = self.fpath	
    data_csv = os.path.join(fpath, 'data.csv')
    df = pd.read_csv(data_csv)

    return df

class PairedDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        drug, target, label = data['smiles'], data['sequence'], data['label']
        drug = smiles2int(drug)
        if len(drug) < 100:
            drug = np.pad(drug, (0, 100 - len(drug)))
        else:
            drug = drug[:100]

        target = seqs2int(target)
        if len(target) < 1200:
            target = np.pad(target, (0, 1200 - len(target)))
        else:
            target = target[:1200]

        return torch.tensor(drug, dtype=torch.long), torch.tensor(target, dtype=torch.long), torch.tensor(label, dtype=torch.float)




# %%
