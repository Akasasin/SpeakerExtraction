from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from .AudioReader import AudioReader
from .LabelReader import LabelReader
import torch.nn.functional as F
import random


def make_dataloader(is_train=True,
                    data_kwargs=None,
                    num_workers=4,
                    chunk_size=32000,
                    batch_size=16,
                    labels_txt=None):
    dataset = Datasets(**data_kwargs, labels_txt=labels_txt)
    return DataLoaders(dataset,
                       is_train=is_train,
                       chunk_size=chunk_size,
                       batch_size=batch_size,
                       num_workers=num_workers)


class Datasets(Dataset):
    """
       Load audio data
       mix_scp: file path of mix audio (type: str)
       ref_scp: file path of ground truth audio (type: list[spk1,spk2])
    """

    def __init__(self, mix_scp=None, ref_scp=None, tar_scp=None, labels_txt=None, sr=8000):
        super(Datasets, self).__init__()
        self.mix_audio = AudioReader(mix_scp, samplerate=sr)
        self.ref_audio = AudioReader(ref_scp, samplerate=sr)
        self.tar_audio = AudioReader(tar_scp, samplerate=sr)
        self.labels = LabelReader(self.mix_audio.keys, labels_txt)

    def __len__(self):
        return len(self.mix_audio)

    def __getitem__(self, index):
        key = self.mix_audio.keys[index]
        mix = self.mix_audio[key]
        ref = self.ref_audio[key]
        tar = self.tar_audio[key]
        return {
            'mix': mix,
            'ref': ref,
            'tar': tar,
            'label': self.labels[index],
            'key': key
        }


class Spliter:
    """
       Split the audio. All audio is divided
       into 4s according to the requirements in the paper.
       input:
             chunk_size: split size
             least: Less than this value will not be read
    """

    def __init__(self, chunk_size=32000, is_train=True, least=16000):
        super(Spliter, self).__init__()
        self.chunk_size = chunk_size
        self.is_train = is_train
        self.least = least

    def chunk_audio(self, sample, start):
        """
           Make a chunk audio
           sample: a audio sample
           start: split start time
        """
        chunk = dict()
        chunk['mix'] = sample['mix'][start:start + self.chunk_size]
        chunk['ref'] = sample['ref'][start:start + self.chunk_size]
        chunk['tar'] = sample['tar'][start:start + self.chunk_size]
        chunk['label'] = sample['label']
        chunk['key'] = sample['key']
        return chunk

    def splits(self, sample):
        """
           Split a audio sample
        """
        length = sample['mix'].shape[0]
        if length < self.least:
            return []
        audio_lists = []
        if length < self.chunk_size:
            gap = self.chunk_size - length
            sample['mix'] = F.pad(sample['mix'], (0, gap), mode='constant')
            sample['ref'] = F.pad(sample['ref'], (0, gap), mode='constant')
            sample['tar'] = F.pad(sample['tar'], (0, gap), mode='constant')
            audio_lists.append(sample)
        else:
            random_start = random.randint(0, length % self.least) if self.is_train else 0
            while True:
                if random_start + self.chunk_size > length:
                    break
                audio_lists.append(self.chunk_audio(sample, random_start))
                random_start += self.least
        return audio_lists


class DataLoaders:
    """
        Custom dataloader method
        input:
              dataset (Dataset): dataset from which to load the data.
              num_workers (int, optional): how many subprocesses to use for data (default: 4)
              chunk_size (int, optional): split audio size (default: 32000(4 s))
              batch_size (int, optional): how many samples per batch to load
              is_train: if this dataloader for training
    """

    def __init__(self, dataset, num_workers=4, chunk_size=32000, batch_size=1, is_train=True):
        super(DataLoaders, self).__init__()
        self.dataset = dataset
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.is_train = is_train
        self.data_loader = DataLoader(self.dataset,
                                      num_workers=self.num_workers,
                                      batch_size=self.batch_size,
                                      shuffle=self.is_train,
                                      collate_fn=self._collate)
        self.spliter = Spliter(chunk_size=self.chunk_size, is_train=self.is_train, least=self.chunk_size // 2)

    def _collate(self, batch):
        """
            merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        """
        batch_audio = []
        for b in batch:
            batch_audio += self.spliter.splits(b)
        return batch_audio

    def __iter__(self):
        mini_batch = []
        for batch in self.data_loader:
            mini_batch += batch
            length = len(mini_batch)
            if self.is_train:
                random.shuffle(mini_batch)
            collate_chunk = []
            for start in range(0, length - self.batch_size + 1, self.batch_size):
                b = default_collate(mini_batch[start:start + self.batch_size])
                collate_chunk.append(b)
            idx = length % self.batch_size
            mini_batch = mini_batch[-idx:] if idx else []
            for m_batch in collate_chunk:
                yield m_batch  # batch of datasets
                '''
                   mini_batch like this
                   'mix': batch x L
                   'ref': batch x L
                   'tar': [batch x L, batch x L]
                '''


if __name__ == "__main__":
    kwargs = {
        'mix_scp': '../datasets/scp/tr_mix.scp',
        'ref_scp': '../datasets/scp/tr_ref.scp',
        'tar_scp': '../datasets/scp/tr_tar.scp',
        'sr': 8000
    }
    labels_txt = '../datasets/scp/labels.txt'

    train_loader = make_dataloader(is_train=True, data_kwargs=kwargs, num_workers=1, chunk_size=32000, batch_size=1, labels_txt=labels_txt)
    for eg in train_loader:
        mix = eg['mix']
        ref = eg['ref']
        tar = eg['tar']
        label = eg['label']
        key = eg['key']
        print(mix.shape)
        print(ref.shape)
        print(tar.shape)
        print(label)
        print(key)
        break
