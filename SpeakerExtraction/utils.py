import torch
import yaml
from torch.autograd import Variable


def parse(opt_path):
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    opt['resume']['path'] = opt['resume']['path'] + '/' + opt['name']
    opt['logger']['path'] = opt['logger']['path'] + '/' + opt['name']
    return opt


def handle_scp(scp_path):
    scp_dict = dict()
    line = 0
    lines = open(scp_path, 'r').readlines()
    for l in lines:
        scp_parts = l.strip().split()
        line += 1
        if len(scp_parts) != 2:
            raise RuntimeError("For {}, format error in line[{:d}]: {}".format(scp_path, line, scp_parts))
        if len(scp_parts) == 2:
            key, value = scp_parts
            if key in scp_dict:
                raise ValueError("Duplicated key \'{0}\' exists in {1}".format(key, scp_path))
            scp_dict[key] = value

    return scp_dict


def check_parameters(net):
    """
        Returns module parameters. Mb
    """
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10 ** 6


def pad_segment(x, segment_size):
    # input is the features: (B, N, T)
    batch_size, dim, seq_len = x.shape
    segment_stride = segment_size // 2
    rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
    if rest > 0:
        pad = Variable(torch.zeros(batch_size, dim, rest)).type(x.type())
        x = torch.cat([x, pad], 2)

    pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(x.type())
    x = torch.cat([pad_aux, x, pad_aux], 2)
    return x, rest


def create_chunks(x, segment_size):
    # split the feature into chunks of segment size
    # input is the features: (B, N, T)

    x, rest = pad_segment(x, segment_size)
    batch_size, dim, seq_len = x.shape
    segment_stride = segment_size // 2

    segments1 = x[:, :, :-segment_stride].contiguous().view(batch_size,
                                                            dim, -1, segment_size)
    segments2 = x[:, :, segment_stride:].contiguous().view(
        batch_size, dim, -1, segment_size)
    segments = torch.cat([segments1, segments2], 3).view(batch_size, dim, -1, segment_size).transpose(2, 3)
    return segments.contiguous(), rest


def merge_chunks(x, rest):
    # merge the split features into full utterance
    # input is the features: (B, N, L, K)

    batch_size, dim, segment_size, _ = x.shape
    segment_stride = segment_size // 2
    x = x.transpose(2, 3).contiguous().view(batch_size, dim, -1, segment_size * 2)  # B, N, K, L

    input1 = x[:, :, :, :segment_size].contiguous().view(batch_size, dim, -1)[:, :, segment_stride:]
    input2 = x[:, :, :, segment_size:].contiguous().view(batch_size, dim, -1)[:, :, :-segment_stride]

    output = input1 + input2
    if rest > 0:
        output = output[:, :, :-rest]
    return output.contiguous()  # B, N, T


if __name__ == '__main__':
    path = './datasets/scp/cv_mix.scp'
    result = handle_scp(path)
    print(len(result))
    print(result[0].get('key'))
    print(result[0].get('value'))
