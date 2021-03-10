# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import math
import time
import numpy as np
import pickle
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import pandas as pd

from dataset import *
from network.ML_DTI import DTImodel
from utils import *
from config.config_dict import Config
from log.train_logger import TrainLogger

# %%
def get_cindex(Y, P):
    summ = 0.
    pair = 0
    
    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if(Y[i] > Y[j]):
                    pair += 1
                    summ +=  1* (P[i] > P[j]) + 0.5 * (P[i] == P[j])
        
    if pair != 0:
        return summ/pair, pair
    else:
        return 0, pair

def val(model, criterion, dataloader):
    model.eval()
    running_loss = AverageMeter()
    running_cindex = AverageMeter()

    for data in dataloader:
        drug, target, label = data
        drug, target, label = drug.cuda(), target.cuda(), label.cuda()

        with torch.no_grad():
            pred = model(target, drug)
            loss = criterion(pred.view(-1), label)
            cindex, pair = get_cindex(label.detach().cpu().numpy().reshape(-1), pred.detach().cpu().numpy().reshape(-1))

            running_loss.update(loss.item(), label.size(0))
            running_cindex.update(cindex, pair)

    epoch_loss = running_loss.get_average()
    epoch_cindex = running_cindex.get_average()
    running_loss.reset()
    running_cindex.reset()

    model.train()

    return epoch_loss, epoch_cindex

# %%
for fold in range(5):
    config = Config()
    args = config.get_config()
    args['fold'] = fold
    logger = TrainLogger(args)
    logger.info(__file__)

    data_root = args.get("data_root")
    DATASET = args.get("dataset")
    split_type = args.get("split_type")
    save_model = args.get("save_model")
    fold = args.get("fold")

    fpath = os.path.join(data_root, DATASET)
    dp = DataPrepared(fpath)
    train_index, val_index, test_index = dp.read_sets(fold, split_type=split_type)
    df = dp.get_data()
    train_df = df.iloc[train_index]
    val_df = df.iloc[val_index]
    test_df = df.iloc[test_index]

    train_set = PairedDataset(train_df)
    val_set = PairedDataset(val_df)
    test_set = PairedDataset(test_df)

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)

    model = DTImodel(dp.vocab_protein_len + 1, dp.vocab_ligand_len + 1).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    epochs = 1000
    steps_per_epoch = 200
    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))
    break_flag = False

    global_step = 0
    global_epoch = 0
    early_stop_epoch = 50

    running_loss = AverageMeter()
    running_cindex = AverageMeter()
    running_best_mse = BestMeter("min")

    model.train()

    for i in range(num_iter):
        if break_flag:
            break

        for data in train_loader:
            global_step += 1
            drug, target, label = data
            drug, target, label = drug.cuda(), target.cuda(), label.cuda()

            pred = model(target, drug)

            loss = criterion(pred.view(-1), label)
            cindex, pair = get_cindex(label.detach().cpu().numpy().reshape(-1), pred.detach().cpu().numpy().reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(loss.item(), label.size(0)) 
            running_cindex.update(cindex, pair)
            
            if global_step % steps_per_epoch == 0:
                global_epoch += 1

                epoch_loss = running_loss.get_average()
                epoch_cindex = running_cindex.get_average()
                running_loss.reset()
                running_cindex.reset()

                val_loss, val_cindex = val(model, criterion, val_loader)
                test_loss, test_cindex = val(model, criterion, test_loader)

                msg = "epoch-%d, loss-%.4f, cindex-%.4f, val_loss-%.4f, val_cindex-%.4f, test_loss-%.4f, test_cindex-%.4f" % (global_epoch, epoch_loss, epoch_cindex, val_loss, val_cindex, test_loss, test_cindex)
                logger.info(msg)

                if save_model:

                    save_model_dict(model, logger.get_model_dir(), msg)

                    if val_loss < running_best_mse.get_best():
                        running_best_mse.update(val_loss)
                    else:
                        count = running_best_mse.counter()
                        if count > early_stop_epoch:
                            logger.info(f"early stop in epoch {global_epoch}")
                            break_flag = True
                            break
    
            

# %%


