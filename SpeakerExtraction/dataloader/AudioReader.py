import torch
import soundfile
import librosa
# from ..utils import handle_scp


def handle_scp(scp_path):
    scp_dict = dict()
    line = 0
    lines = open(scp_path, 'r').readlines()
    print("original path: ", lines)
    for l in lines:
        scp_parts = l.strip().split()
        line += 1
        if len(scp_parts) != 2:
            raise RuntimeError("For {}, format error in line[{:d}]: {}".format(scp_path, line, scp_parts))
        if len(scp_parts) == 2:
            key, value = scp_parts
            if key in scp_dict:
                raise ValueError("Duplicated key \'{0}\' exists in {1}".format(key, scp_path))
            scp_dict[key] = value.replace("\\", "/")

    return scp_dict

def read_wav(filepath, samplerate):
    # print("the input libsora path is : ", filepath)
    src, _ = librosa.load(filepath, sr=samplerate)
    # print("libsora reading donw....")
    src = torch.tensor(src, dtype=torch.float32).squeeze()
    return src


def write_wav(filepath, src, samplerate):
    soundfile.write(filepath, src.cpu().numpy(), samplerate)


class AudioReader(object):
    def __init__(self, scp_path, samplerate=8000):
        super(AudioReader, self).__init__()
        self.samplerate = samplerate
        self.index_dict = handle_scp(scp_path)
        print("index_dict for libsora； ", self.index_dict)
        self.keys = list(self.index_dict.keys())

    def _load(self, key):
        print("self.index_dict[key]: ", self.index_dict[key])
        print("All input file path: ", self.index_dict)
        src = read_wav(self.index_dict[key], samplerate=self.samplerate)
        return src

    def __len__(self):
        return len(self.keys)

    def __iter__(self):
        for key in self.keys:
            yield key, self._load(key)

    def __getitem__(self, index):
        if type(index) not in [int, str]:
            raise IndexError('Unsupported index type: {}'.format(type(index)))
        if type(index) is int:
            num_uttrs = len(self.keys)
            if num_uttrs < index < 0:
                raise KeyError('Integer index out of range, {:d} vs {:d}'.format(index, num_uttrs))
            index = self.keys[index]
        if index not in self.index_dict:
            raise KeyError("Missing utterance {}!".format(index))

        return self._load(index)


if __name__ == "__main__":
    r = AudioReader('../datasets/minimum_noise/cv_s2.scp')
    print(r[0].shape)
    print(r[1])
