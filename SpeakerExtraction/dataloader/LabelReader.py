import torch


class LabelReader(object):
    def __init__(self, keys, labels_txt):
        super().__init__()
        self.keys = keys
        table = self.handle_txt(labels_txt)
        self.labels = torch.zeros(len(keys), dtype=torch.int64)
        for i, key in enumerate(keys):
            self.labels[i] = table[key[7:12]]

    def __len__(self):
        return len(self.keys)

    def __iter__(self):
        for i in range(len(self.keys)):
            yield self.keys[i], self.labels[i]

    def __getitem__(self, index):
        return self.labels[index]

    @staticmethod
    def handle_txt(labels_txt):
        table = dict()
        with open(labels_txt, 'r') as f:
            idx = 0
            for line in f.readlines():
                tup = line.split('\t')
                key, val = tup[0].strip(), idx
                idx = idx + 1
                table[key] = val
        return table
