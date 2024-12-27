import torch
import torch.nn as nn
import torch.nn.functional as F
from util import check_parameters, create_chunks, merge_chunks
from models.conformer.Conformer import MHSAModule


class SpeechEncoder(nn.Module):
    def __init__(self, N=256, L1=20, L2=80, L3=160):
        super().__init__()
        self.conv_s = nn.Conv1d(1, N, L1, L1 // 2, bias=False)
        self.conv_m = nn.Conv1d(1, N, L2, L1 // 2, (L2 - L1) // 2, bias=False)
        self.conv_l = nn.Conv1d(1, N, L3, L1 // 2, (L3 - L1) // 2, bias=False)

    def forward(self, x):
        x = x.unsqueeze(1)
        x_s = F.relu(self.conv_s(x))
        x_m = F.relu(self.conv_m(x))
        x_l = F.relu(self.conv_l(x))
        x = torch.cat([x_s, x_m, x_l], dim=1)
        return x_s, x_m, x_l, x


class SpeechDecoder(nn.Module):
    def __init__(self, N=256, L1=20, L2=80, L3=160):
        super().__init__()
        self.convT_s = nn.ConvTranspose1d(N, 1, L1, L1 // 2, bias=False)
        self.convT_m = nn.ConvTranspose1d(N, 1, L2, L1 // 2, (L2 - L1) // 2, bias=False)
        self.convT_l = nn.ConvTranspose1d(N, 1, L3, L1 // 2, (L3 - L1) // 2, bias=False)

    def forward(self, x_s, x_m, x_l):
        x_s = self.convT_s(x_s).squeeze(1)
        x_m = self.convT_m(x_m).squeeze(1)
        x_l = self.convT_l(x_l).squeeze(1)
        return x_s, x_m, x_l


class ResNetBlock(nn.Module):
    def __init__(self, N=256):
        super().__init__()
        self.conv1 = nn.Conv1d(N, N, 1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(N)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv1d(N, N, 1, bias=False)
        self.batch_norm2 = nn.BatchNorm1d(N)
        self.prelu2 = nn.PReLU()
        self.max_pool = nn.MaxPool1d(3, 1, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.prelu1(self.batch_norm1(y))
        y = self.conv2(y)
        x = self.batch_norm2(y) + x
        x = self.max_pool(self.prelu2(x))
        return x


class SpeakerClassifier(nn.Module):
    def __init__(self, N=256, Nr=3, label_length=100):
        super().__init__()
        self.label_length = label_length
        self.batch_norm = nn.BatchNorm1d(3 * N)
        self.conv1 = nn.Conv1d(3 * N, N, 1, bias=False)
        self.resnet_blocks = nn.ModuleList([ResNetBlock(N) for _ in range(Nr)])
        self.conv2 = nn.Conv1d(N, N, 1, bias=False)
        self.mean_pooling = nn.AvgPool1d(3, 1, 1)
        self.softmax = nn.Softmax(dim=1)

    # input [B, N, T]
    def forward(self, x):
        x = self.batch_norm(x)
        x = self.conv1(x)
        for resnet_block in self.resnet_blocks:
            x = resnet_block(x)
        x = self.conv2(x)
        v = self.mean_pooling(x)
        B, N, T = v.shape
        x = v.contiguous().view(B, N * T)
        linear = nn.Linear(N * T, self.label_length, device=x.device)
        pred = self.softmax(linear(x))
        return v, pred


class ConvModule(nn.Module):
    def __init__(self, N, H, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(N)
        self.linear = nn.Linear(N, H)
        self.dw_conv = nn.Conv1d(H, H, 31, padding=15, groups=H, dilation=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.silu(self.linear(x)).transpose(1, 2)
        x = self.dw_conv(x) + x
        x = self.dropout(x.transpose(1, 2))
        return x


class DPConvAtt(nn.Module):
    def __init__(self, N, num_heads, dropout):
        super().__init__()
        self.conv1x1 = nn.Conv2d(2 * N, N, 1, bias=False)
        self.intra_convM1 = ConvModule(N, N, dropout)
        self.intra_attention = MHSAModule(N, num_heads, dropout)
        self.intra_convM2 = ConvModule(N, N, dropout)
        self.inter_convM1 = ConvModule(N, N, dropout)
        self.inter_attention = MHSAModule(N, num_heads, dropout)
        self.inter_convM2 = ConvModule(N, N, dropout)

    # x: [B, N, K, S], v: [B, N, K, S]
    def forward(self, x, v):
        B, N, K, S = x.shape
        y = torch.cat([x, v], 1)
        y = self.conv1x1(y)
        intra = y.permute(0, 3, 2, 1).contiguous().view(B * S, K, N)
        intra = intra + self.intra_convM1(intra)
        intra = intra + self.intra_attention(intra)
        intra = intra + self.intra_convM2(intra)
        x = intra.view(B, S, K, N).permute(0, 3, 2, 1).contiguous() + x
        inter = x.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)
        inter = inter + self.inter_convM1(inter)
        inter = inter + self.inter_attention(inter)
        inter = inter + self.inter_convM2(inter)
        x = inter.view(B, K, S, N).permute(0, 3, 1, 2).contiguous() + x
        return x


class Separator(nn.Module):
    def __init__(self, N=256, heads=4, R=4, segment_size=80, dropout=0.0):
        super().__init__()
        self.segment_size = segment_size
        self.layer_norm = nn.LayerNorm(3 * N)
        self.conv1x1 = nn.Conv1d(3 * N, N, 1, bias=False)
        self.conformer_blocks = nn.ModuleList([DPConvAtt(N, heads, dropout) for _ in range(R)])
        self.conv_s = nn.Sequential(nn.Conv1d(N, N, 1, bias=False), nn.ReLU())
        self.conv_m = nn.Sequential(nn.Conv1d(N, N, 1, bias=False), nn.ReLU())
        self.conv_l = nn.Sequential(nn.Conv1d(N, N, 1, bias=False), nn.ReLU())

    def forward(self, x, v):
        x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.conv1x1(x)
        # [B, N, K, S]
        x, rest = create_chunks(x, self.segment_size)
        v, _ = create_chunks(v, self.segment_size)
        for block in self.conformer_blocks:
            x = block(x, v)
        # [B, N, T]
        x = merge_chunks(x, rest)
        m_s = self.conv_s(x)
        m_m = self.conv_m(x)
        m_l = self.conv_l(x)
        return m_s, m_m, m_l


class SpExDPConformer(nn.Module):
    def __init__(self, N=256, L1=20, L2=80, L3=160, heads=4, Nr=3, R=8, segment_size=80, dropout=0.0, label_length=100):
        super().__init__()
        self.encoder = SpeechEncoder(N, L1, L2, L3)
        self.decoder = SpeechDecoder(N, L1, L2, L3)
        self.classifier = SpeakerClassifier(N, Nr, label_length)
        self.separator = Separator(N, heads, R, segment_size, dropout)

    def forward(self, mix, aux):
        y_s, y_m, y_l, y = self.encoder(mix)
        _, _, _, x = self.encoder(aux)
        v, pred = self.classifier(x)
        m_s, m_m, m_l = self.separator(y, v)
        o_s, o_m, o_l = self.decoder(y_s * m_s, y_m * m_m, y_l * m_l)
        return o_s, o_m, o_l, pred


if __name__ == '__main__':
    net = SpExDPConformer().cuda()
    print(f'model parameters: {check_parameters(net)}M')
    mix = torch.randn([2, 32000], dtype=torch.float32).cuda()
    aux = torch.randn([2, 32000], dtype=torch.float32).cuda()
    o_s, o_m, o_l, pred = net(mix, aux)
    print(o_s.shape, pred.shape)
