# %%
import torch.nn as nn
import torch
from torch.nn.modules import padding
from torch.nn.modules.conv import Conv1d
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

class Conv1dReLU(nn.Module):
    '''
    kernel_size=3, stride=1, padding=1
    kernel_size=5, stride=1, padding=2
    kernel_size=7, stride=1, padding=3
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size,  stride=stride, padding=padding),
            nn.ReLU()
        )
    
    def forward(self, x):

        return self.inc(x)

class LinearReLU(nn.Module):
    def __init__(self,in_features, out_features, bias=True):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
            nn.ReLU()
        )

    def forward(self, x):
        
        return self.inc(x)

class BilinearPooling(nn.Module):
    def __init__(self, in_channels, out_channels, c_m, c_n):
        super().__init__()

        self.convA = nn.Conv1d(in_channels, c_m, kernel_size=1, stride=1, padding=0)
        self.convB = nn.Conv1d(in_channels, c_n, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(c_m, out_channels, bias=True)

    def forward(self, x):
        '''
        x: (batch, channels, seq_len)
        A: (batch, c_m, seq_len)
        B: (batch, c_n, seq_len)
        att_maps.permute(0, 2, 1): (batch, seq_len, c_n)
        global_descriptors: (batch, c_m, c_n)
        '''        
        A = self.convA(x) 
        B = self.convB(x)
        att_maps = F.softmax(B, dim=-1)
        global_descriptors = torch.bmm(A, att_maps.permute(0, 2, 1))
        global_descriptor = torch.mean(global_descriptors, dim=-1)
        out = self.linear(global_descriptor).unsqueeze(1)

        return out

class MutualAttentation(nn.Module):
    def __init__(self, in_channels, att_size, c_m, c_n):
        super().__init__()
        self.bipool = BilinearPooling(in_channels, in_channels, c_m, c_n)
        self.linearS = nn.Linear(in_channels, att_size)
        self.linearT = nn.Linear(in_channels, att_size)
    
    def forward(self, source, target):
        '''
        source: (batch, channels, seq_len)
        target: (batch, channels, seq_len)
        global_descriptor: (batch, 1, channels)
        '''
        global_descriptor = self.bipool(source)
        target_org = target
        target = self.linearT(target.permute(0, 2, 1)).permute(0, 2, 1)
        global_descriptor = self.linearS(global_descriptor)
        att_maps = torch.bmm(global_descriptor, target)
        att_maps = F.sigmoid(att_maps)
        out_target = torch.add(target_org, torch.mul(target_org, att_maps))
        out_target = F.relu(out_target)

        return out_target


class DTImodel(nn.Module):
    def __init__(self, vocab_prot_size, vocab_drug_size, embedding_size=128, filter_num=32):
        super().__init__()
        prot_filter_size = [4, 8, 12]
        drug_filter_size = [4, 6, 8]

        self.prot_embed = nn.Embedding(vocab_prot_size, embedding_size, padding_idx=0)
        self.prot_conv1 = Conv1dReLU(embedding_size, filter_num, prot_filter_size[0])
        self.prot_conv2 = Conv1dReLU(filter_num, filter_num * 2, prot_filter_size[1])
        self.prot_conv3 = Conv1dReLU(filter_num * 2, filter_num * 3, prot_filter_size[2])
        self.prot_pool = nn.AdaptiveMaxPool1d(1)

        self.drug_embed = nn.Embedding(vocab_drug_size, embedding_size, padding_idx=0)
        self.drug_conv1 = Conv1dReLU(embedding_size, filter_num, drug_filter_size[0])
        self.drug_conv2 = Conv1dReLU(filter_num, filter_num * 2, drug_filter_size[1])
        self.drug_conv3 = Conv1dReLU(filter_num * 2, filter_num * 3, drug_filter_size[2])
        self.drug_pool = nn.AdaptiveMaxPool1d(1)

        self.prot_mut_att1 = MutualAttentation(filter_num, filter_num, filter_num, 8)
        self.drug_mut_att1 = MutualAttentation(filter_num, filter_num, filter_num, 8)

        self.prot_mut_att2 = MutualAttentation(filter_num*2, filter_num, filter_num, 8)
        self.drug_mut_att2 = MutualAttentation(filter_num*2, filter_num, filter_num, 8)

        self.prot_mut_att3 = MutualAttentation(filter_num*3, filter_num, filter_num, 8)
        self.drug_mut_att3 = MutualAttentation(filter_num*3, filter_num, filter_num, 8)

        self.linear1 = LinearReLU(filter_num * 3 * 2, 1024)
        self.drop1 = nn.Dropout(0.1)
        self.linear2 = LinearReLU(1024, 1024)
        self.drop2 = nn.Dropout(0.1)
        self.linear3 = LinearReLU(1024, 512)
        self.drop3 = nn.Dropout(0.1)
        self.out_layer = nn.Linear(512, 1)

    def forward(self, prot_x, drug_x):
        prot_x = self.prot_embed(prot_x).permute(0, 2, 1)
        drug_x = self.drug_embed(drug_x).permute(0, 2, 1)

        prot_x = self.prot_conv1(prot_x)
        drug_x = self.drug_conv1(drug_x)
        prot_x_g = self.prot_mut_att1(drug_x, prot_x)
        drug_x_g = self.drug_mut_att1(prot_x, drug_x)

        prot_x = self.prot_conv2(prot_x_g)
        drug_x = self.drug_conv2(drug_x_g)
        prot_x_g = self.prot_mut_att2(drug_x, prot_x)
        drug_x_g = self.drug_mut_att2(prot_x, drug_x)

        prot_x = self.prot_conv3(prot_x_g)
        drug_x = self.drug_conv3(drug_x_g)
        prot_x_g = self.prot_mut_att3(drug_x, prot_x)
        drug_x_g = self.drug_mut_att3(prot_x, drug_x)

        prot_x = self.prot_pool(prot_x_g).squeeze(-1)
        drug_x = self.drug_pool(drug_x_g).squeeze(-1)

        x = torch.cat([prot_x, drug_x], dim=-1)
        x = self.linear1(x)
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        x = self.linear3(x)
        x = self.drop3(x)
        x = self.out_layer(x)

        return x

# %%
if __name__ == "__main__":
    protein_x = torch.randint(0, 20, (1, 1200))
    ligand_x = torch.randint(0, 65, (1, 64))   
    net = DTImodel(20+1, 65+1)
    res = net(protein_x, ligand_x)
    print(res.shape)

# %%