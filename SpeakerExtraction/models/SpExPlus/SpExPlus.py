import torch
import torch.nn as nn
import torch.nn.functional as F
# from ...utils import check_parameters

def check_parameters(net):
    """
        Returns module parameters. Mb
    """
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10 ** 6


class GlobalLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.dim, 1))
            self.bias = nn.Parameter(torch.zeros(self.dim, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))

        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x-mean)**2, (1, 2), keepdim=True)
        # N x C x L
        if self.elementwise_affine:
            x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
        else:
            x = (x-mean)/torch.sqrt(var+self.eps)
        return x


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


class Conv1D_Block(nn.Module):
    def __init__(self, N=256, O=256, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(N, O, 1, bias=False)
        self.prelu1 = nn.PReLU()
        self.gln1 = GlobalLayerNorm(O, elementwise_affine=True)
        self.pad = (dilation * (3 - 1)) // 2
        self.dw_conv = nn.Conv1d(O, O, 3, groups=O, padding=self.pad, dilation=dilation)
        self.prelu2 = nn.PReLU()
        self.gln2 = GlobalLayerNorm(O, elementwise_affine=True)
        self.conv2 = nn.Conv1d(O, N, 1, bias=False)

    def forward(self, x):
        y = self.conv1(x)
        y = self.gln1(self.prelu1(y))
        y = self.dw_conv(y)
        y = self.gln2(self.prelu2(y))
        x = self.conv2(y) + x
        return x


class TCNBlock(nn.Module):
    def __init__(self, N=256, O=256, B=8):
        super().__init__()
        self.conv1 = nn.Conv1d(2 * N, O, 1, bias=False)
        self.prelu1 = nn.PReLU()
        self.gln1 = GlobalLayerNorm(O, elementwise_affine=True)
        self.dw_conv = nn.Conv1d(O, O, 3, groups=O, padding=1, dilation=1)
        self.prelu2 = nn.PReLU()
        self.gln2 = GlobalLayerNorm(O, elementwise_affine=True)
        self.conv2 = nn.Conv1d(O, N, 1, bias=False)
        self.blocks = nn.ModuleList([Conv1D_Block(N, O, dilation=(2**i)) for i in range(1, B)])

    def forward(self, x, v):
        y = torch.cat([x, v], dim=1)
        y = self.conv1(y)
        y = self.gln1(self.prelu1(y))
        y = self.dw_conv(y)
        y = self.gln2(self.prelu2(y))
        x = self.conv2(y) + x
        for block in self.blocks:
            x = block(x)
        return x


class SpeakerClassifier(nn.Module):
    def __init__(self, N=256, Nr=3, label_length=212):
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
        x = self.softmax(linear(x))
        return v, x


class SpeechSeparator(nn.Module):
    def __init__(self, N=256, O=256, B=8, R=4):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(3 * N)
        self.conv1x1 = nn.Conv1d(3 * N, N, 1, bias=False)
        self.tcn_blocks = nn.ModuleList([TCNBlock(N, O, B) for _ in range(R)])
        self.conv_s = nn.Conv1d(N, N, 1, bias=False)
        self.conv_m = nn.Conv1d(N, N, 1, bias=False)
        self.conv_l = nn.Conv1d(N, N, 1, bias=False)

    def forward(self, x, v):
        x = self.batch_norm(x)
        x = self.conv1x1(x)
        for tcn_block in self.tcn_blocks:
            x = tcn_block(x, v)

        mask_s = F.relu(self.conv_s(x))
        mask_m = F.relu(self.conv_m(x))
        mask_l = F.relu(self.conv_l(x))

        return mask_s, mask_m, mask_l


class SpExPlus(nn.Module):
    def __init__(self, N=256, L1=20, L2=80, L3=160, O=256, Nr=3, B=8, R=4, label_length=51):
        super().__init__()
        self.encoder = SpeechEncoder(N, L1, L2, L3)
        self.speaker_classifier = SpeakerClassifier(N, Nr, label_length)
        self.speech_separator = SpeechSeparator(N, O, B, R)
        self.decoder = SpeechDecoder(N, L1, L2, L3)

    def forward(self, mix, aux):
        _, _, _, x = self.encoder(aux)
        y_s, y_m, y_l, y = self.encoder(mix)
        v, spk_pred = self.speaker_classifier(x)
        mask_s, mask_m, mask_l = self.speech_separator(y, v)
        out_s, out_m, out_l = self.decoder(mask_s * y_s, mask_m * y_m, mask_l * y_l)
        return out_s, out_m, out_l, spk_pred


if __name__ == '__main__':
    mix = torch.randn([1, 32000], dtype=torch.float32).cuda()
    aux = torch.randn([1, 32000], dtype=torch.float32).cuda()
    net = SpExPlus().cuda()
    print(f'model parameters: {check_parameters(net)}')
    out_s, out_m, out_l, spk_pred = net(mix, aux)
    print(out_s.shape)
    print(out_m.shape)
    print(out_l.shape)
    print(spk_pred)
