# %%
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, pad_packed_sequence
import numpy as np
from dataset import *

# %%
class DrugEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size=128, hidden_size=96):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.encoder = nn.LSTM(embedding_size, hidden_size, num_layers=3, batch_first=True)
    
    def pack_sequence(self, x):
        lengths = [torch.sum(d != 0) for d in x]
        lengths, indices = torch.sort(torch.tensor(lengths), descending=True)
        x = x[indices]

        return x, lengths, indices

    def forward(self, x):
        x, lengths, indices = self.pack_sequence(x)
        x = self.embed(x)
        x = pack_padded_sequence(x, lengths, batch_first=True)
        encoder_outputs_packed, (h_last, c_last) = self.encoder(x)
        h = h_last[-1]
        h[indices] = h[range(indices.size(0))]

        return h

        
class TargetEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, kernel_sizes, filter_num=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.conv1 = nn.Conv1d(embedding_size, filter_num, kernel_size=kernel_sizes[0], stride=1)
        self.conv2 = nn.Conv1d(filter_num, filter_num * 2, kernel_size=kernel_sizes[1], stride=1)
        self.conv3 = nn.Conv1d(filter_num * 2, filter_num * 3, kernel_size=kernel_sizes[2], stride=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        
    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)

        return x

class DTImodel(nn.Module):
    def __init__(self, vocab_protein_size, vocab_ligand_size, embedding_size=128, filter_num=32):
        super().__init__()
        self.protein_encoder = TargetEncoder(vocab_protein_size, embedding_size, kernel_sizes=[4, 8, 12], filter_num=filter_num)
        self.ligand_encoder = DrugEncoder(vocab_ligand_size, embedding_size)
        self.linear1 = nn.Linear(filter_num * 3 * 2, 1024)
        self.drop1 = nn.Dropout(0.1)
        self.linear2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout(0.1)
        self.linear3 = nn.Linear(1024, 512)
        self.drop3 = nn.Dropout(0.1)
        self.out_layer = nn.Linear(512, 1)

    def forward(self, protein_x, ligand_x):
        protein_x = self.protein_encoder(protein_x)
        ligand_x = self.ligand_encoder(ligand_x)
        x = torch.cat([protein_x, ligand_x], dim=-1)
        x = F.relu(self.linear1(x))
        x = self.drop1(x)
        x = F.relu(self.linear2(x))
        x = self.drop2(x)
        x = F.relu(self.linear3(x))
        x = self.drop3(x)
        x = self.out_layer(x)

        return x


if __name__ == "__main__":
    DATASET = 'davis'
    fpath = os.path.join(r'..\preprocessed_data', DATASET)
    dp = DataPrepared(fpath)
    train_index, val_index, test_index = dp.read_sets(0, split_type='warm')
    df = dp.get_data()
    train_df = df.iloc[train_index]
    val_df = df.iloc[val_index]
    test_df = df.iloc[test_index]

    train_set = PairedDataset(train_df)
    val_set = PairedDataset(val_df)
    test_set = PairedDataset(test_df)

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    data = iter(train_loader).next()
    drug, target, label = data

    # net = DrugEncoder(dp.vocab_ligand_len + 1)
    # res = net(drug)

    net = DTImodel(dp.vocab_protein_len + 1, dp.vocab_ligand_len + 1)
    res = net(target, drug)

# %%
