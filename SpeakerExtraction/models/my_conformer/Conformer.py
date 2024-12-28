import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..Conformer.conformer import Attention, check_parameters
# from ...utils import check_parameters



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class FeedForwardModule(nn.Module):
    def __init__(self, N=256, dropout=0.0):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(N),
            nn.Linear(N, N),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(N, N),
            nn.Dropout(p=dropout)
        )

    # [B, T, N]
    def forward(self, x):
        return self.sequential(x) + 0.5 * x


class ConvolutionModule(nn.Module):
    def __init__(self, N=256, dropout=0.0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(N)
        self.p_conv1 = nn.Conv1d(N, 2 * N, 1, bias=False)
        self.d_conv = nn.Conv1d(N, N, 3, padding=1, groups=N, bias=False)
        self.batch_norm = nn.BatchNorm1d(N)
        self.p_conv2 = nn.Conv1d(N, N, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    # [B, T, N]
    def forward(self, x):
        y = self.layer_norm(x).transpose(1, 2)
        y = F.glu(self.p_conv1(y), dim=1)
        y = self.d_conv(y)
        y = self.batch_norm(y)
        y = F.silu(y)
        y = self.p_conv2(y)
        y = self.dropout(y).transpose(1, 2)
        return x + y


class MHSAModule(nn.Module):
    def __init__(self, N=256, heads=4, dropout=0.0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(N)
        self.positional_encoding = PositionalEncoding(N, dropout)
        self.attention = nn.MultiheadAttention(N, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        y = self.layer_norm(x)
        y = self.positional_encoding(y)
        y, _ = self.attention(y, y, y)
        y = self.dropout(y)
        return x + y


class ConformerBlock(nn.Module):
    def __init__(self, N=256, heads=4, dropout=0.1):
        super().__init__()
        self.sequential = nn.Sequential(
            FeedForwardModule(N, dropout),
            ConvolutionModule(N, dropout),
            MHSAModule(N, heads, dropout),
            FeedForwardModule(N, dropout)
        )

    def forward(self, x):
        return self.sequential(x)


if __name__ == '__main__':
    net1 = MHSAModule(256)
    net2 = Attention(256, 4, 64)
    print(f'mine: {check_parameters(net1)}M')
    print(f'others: {check_parameters(net2)}M')
