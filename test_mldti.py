# %%
import os
os.environ['CUDA_ENABLE_DEVICES'] = '3'
import math
import time
import numpy as np
import pickle
import torch.optim as optim
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import pandas as pd

from dataset import *
from network.ML_DTI import DTImodel
from utils import *
from config.config_dict import Config
from log.test_logger import TestLogger

# %%
def get_cindex(Y, P):
    summ = 0
    pair = 0
    
    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if(Y[i] > Y[j]):
                    pair +=1
                    summ +=  1* (P[i] > P[j]) + 0.5 * (P[i] == P[j])
        
    if pair is not 0:
        return summ/pair
    else:
        return 0

def val(model, criterion, dataloader):
    model.eval()
    running_loss = AverageMeter()

    pred_list = []
    label_list = []

    for data in dataloader:
        drug, target, label = data
        drug, target, label = drug.cuda(), target.cuda(), label.cuda()

        with torch.no_grad():
            pred = model(target, drug)
            loss = criterion(pred.view(-1), label)
            pred_list.append(pred.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            running_loss.update(loss.item(), label.size(0))

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    epoch_cindex = get_cindex(label, pred)

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    return epoch_loss, epoch_cindex

# %%
config = Config(train=False)
args = config.get_config()
logger = TestLogger(args)
logger.info(__file__)

data_root = args.get("data_root")
DATASET = args.get("dataset")
split_type = args.get("split_type")
save_model = args.get("save_model")
fold = args.get("fold")

# %%

fpath = os.path.join(data_root, DATASET, 'CNN-CNN')
dp = DataPrepared(fpath)
train_index, val_index, test_index = dp.read_sets(fold, split_type=split_type)
df = dp.get_data()
dataset = PairedDataset(df)

train_sampler = RandomSampler(train_index)
val_sampler = SequentialSampler(val_index)
test_sampler = SequentialSampler(test_index)

train_loader = DataLoader(dataset, batch_size=256, sampler=train_index)
val_loader = DataLoader(dataset, batch_size=256, sampler=val_index)
test_loader = DataLoader(dataset, batch_size=256, sampler=test_index)

# %%
model = DTImodel(dp.vocab_protein_len + 1, dp.vocab_ligand_len + 1).cuda()
criterion = nn.MSELoss()
load_model_dict(model, logger.get_model_path())

epoch_loss, epoch_cindex = val(model, criterion, test_loader)
logger.info("epoch_loss:%.4f, epoch_cindex:%.4f" % (epoch_loss, epoch_cindex))
# %%


