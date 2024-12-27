import torch
import torch.nn as nn
import torch.nn.functional as F
from util import check_parameters, create_chunks, merge_chunks
from models.conformer.Conformer import ConformerBlock


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
        return x_s, x


class SpeechDecoder(nn.Module):
    def __init__(self, N=256, L1=20):
        super().__init__()
        self.convT_s = nn.ConvTranspose1d(N, 1, L1, L1 // 2, bias=False)

    def forward(self, x):
        x = self.convT_s(x).squeeze(1)
        return x


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


class DPConformerBlock(nn.Module):
    def __init__(self, N=256, heads=4, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv2d(2 * N, N, 1, bias=False)
        self.batch_norm = nn.BatchNorm2d(N)
        self.intra_conformer = ConformerBlock(N, heads, dropout)
        self.inter_conformer = ConformerBlock(N, heads, dropout)

    # x: [B, N, K, S], v: [B, N, K, S]
    def forward(self, x, v):
        x = torch.cat([x, v], 1)
        x = self.batch_norm(self.conv(x))
        B, N, K, S = x.shape
        # intra
        intra = x.permute(0, 3, 2, 1).contiguous().view(B * S, K, N)
        intra = self.intra_conformer(intra).view(B, S, K, N)
        intra = intra.permute(0, 3, 2, 1).contiguous()
        # [B, N, K, S]
        x = intra + x
        # inter
        inter = x.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)
        inter = self.inter_conformer(inter).view(B, K, S, N)
        inter = inter.permute(0, 3, 1, 2).contiguous()
        x = inter + x
        return x


class Separator(nn.Module):
    def __init__(self, N=256, heads=4, R=4, segment_size=80, dropout=0.0):
        super().__init__()
        self.segment_size = segment_size
        self.batch_norm = nn.BatchNorm1d(3 * N)
        self.conv1x1 = nn.Conv1d(3 * N, N, 1, bias=False)
        self.conformer_blocks = nn.ModuleList([DPConformerBlock(N, heads, dropout) for _ in range(R)])
        self.gate1 = nn.Sequential(nn.Conv1d(N, N, 1, bias=False), nn.Tanh())
        self.gate2 = nn.Sequential(nn.Conv1d(N, N, 1, bias=False), nn.Sigmoid())
        self.gen_mask = nn.Sequential(nn.Conv1d(N, N, 1, bias=False), nn.ReLU())

    def forward(self, x, v):
        x = self.batch_norm(x)
        x = self.conv1x1(x)
        # [B, N, K, S]
        x, rest = create_chunks(x, self.segment_size)
        v, _ = create_chunks(v, self.segment_size)
        for block in self.conformer_blocks:
            x = block(x, v)
        # [B, N, T]
        x = merge_chunks(x, rest)
        x = self.gate1(x) * self.gate2(x)
        mask = self.gen_mask(x)
        return mask


class SpExDPConformer(nn.Module):
    def __init__(self, N=256, L1=20, L2=80, L3=160, heads=4, Nr=3, R=3, segment_size=80, dropout=0.0, label_length=100):
        super().__init__()
        self.encoder = SpeechEncoder(N, L1, L2, L3)
        self.decoder = SpeechDecoder(N, L1)
        self.classifier = SpeakerClassifier(N, Nr, label_length)
        self.separator = Separator(N, heads, R, segment_size, dropout)

    def forward(self, mix, aux):
        y_s, y = self.encoder(mix)
        _, x = self.encoder(aux)
        v, pred = self.classifier(x)
        mask = self.separator(y, v)
        out = self.decoder(y_s * mask)
        return out, pred


if __name__ == '__main__':
    net = SpExDPConformer().cuda()
    print(f'model parameters: {check_parameters(net)}Mb')
    mix = torch.randn([1, 32000], dtype=torch.float32).cuda()
    aux = torch.randn([1, 32000], dtype=torch.float32).cuda()
    out, pred = net(mix, aux)
    print(out.shape, pred.shape)
