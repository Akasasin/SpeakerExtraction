import argparse
import logging
import utils
from ModelHelper import *
from logger import set_logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from trainer import Trainer

# network
from models.SpExPlus.SpExPlus import SpExPlus
from models.SpExDPConformer.SpExDPConformer import SpExDPConformer


def train():
    parser = argparse.ArgumentParser(description='Parameters for training')
    parser.add_argument('--opt', type=str, default='train.yml', help='Path to option YAML file.')
    args = parser.parse_args()
    opt = utils.parse(args.opt)
    set_logger.setup_logger(opt['logger']['name'], opt['logger']['path'], screen=opt['logger']['screen'], tofile=opt['logger']['tofile'])
    logger = logging.getLogger(opt['logger']['name'])

    logger.info('Building the model')

    net = SpExDPConformer(label_length=opt['datasets']['label_length'])

    logger.info("Building the optimizer")
    optimizer = make_optimizer(net.parameters(), opt)

    logger.info('Building the dataloader')
    train_dataloader = make_dataloader(is_train=True, data_kwargs=opt['datasets']['train'],
                                       num_workers=opt['datasets']['num_workers'],
                                       chunk_size=opt['datasets']['chunk_size'],
                                       batch_size=opt['datasets']['batch_size'],
                                       labels_txt=opt['datasets']['labels_txt'])

    valid_dataloader = make_dataloader(is_train=False, data_kwargs=opt['datasets']['val'],
                                       num_workers=opt['datasets']['num_workers'],
                                       chunk_size=opt['datasets']['chunk_size'],
                                       batch_size=opt['datasets']['batch_size'],
                                       labels_txt=opt['datasets']['labels_txt'])

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=opt['scheduler']['factor'],
                                  patience=opt['scheduler']['patience'],verbose=True,
                                  min_lr=opt['scheduler']['min_lr'])

    logger.info('Building the Trainer')
    trainer = Trainer(train_dataloader, valid_dataloader, net, optimizer, scheduler, opt)
    trainer.run()


if __name__ == '__main__':
    train()
