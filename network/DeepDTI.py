# %%
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


class SeqRepresentation(nn.Module):
    def __init__(self, vocab_size, embedding_num, kernel_sizes, filter_num=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
        self.conv1 = nn.Conv1d(embedding_num, filter_num, kernel_size=kernel_sizes[0], stride=1)
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
        self.protein_encoder = SeqRepresentation(vocab_protein_size, embedding_size, kernel_sizes=[4, 8, 12], filter_num=filter_num)
        self.ligand_encoder = SeqRepresentation(vocab_ligand_size, embedding_size, kernel_sizes=[4, 6, 8], filter_num=filter_num)
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

# %%
if __name__ == "__main__":
    pass


