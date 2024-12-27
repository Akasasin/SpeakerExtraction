import torch
from dataloader.DataLoaders import Datasets, DataLoaders


def make_optimizer(params, opt):
    optimizer = getattr(torch.optim, opt['optim']['name'])
    if opt['optim']['name'] == 'Adam':
        optimizer = optimizer(params, lr=opt['optim']['lr'], weight_decay=opt['optim']['weight_decay'])
    else:
        optimizer = optimizer(params, lr=opt['optim']['lr'], weight_decay=opt['optim']['weight_decay'],
                              momentum=opt['optim']['momentum'])

    return optimizer


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
