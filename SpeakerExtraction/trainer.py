import logging
import os.path
import time
import torch
from loss import SISNR_Loss, CE_Loss, Accuracy
from utils import check_parameters
from tqdm import tqdm
import matplotlib.pyplot as plt
# from swave.sisnr_loss import cal_loss


class Trainer(object):
    def __init__(self, train_dataloader, valid_dataloader, net, optimizer, scheduler, opt):
        super(Trainer).__init__()
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.scheduler = scheduler
        self.cur_epoch = 0
        self.total_epoch = opt['train']['epoch']
        self.early_stop = opt['train']['early_stop']
        self.print_freq = opt['logger']['print_freq']
        self.logger = logging.getLogger(opt['logger']['name'])
        self.checkpoint = opt['train']['path']
        self.name = opt['name']
        self.alpha, self.beta, self.gamma = opt['train']['alpha'], opt['train']['beta'], opt['train']['gamma']

        if opt['train']['gpuid']:
            self.logger.info('Load Nvidia GPU .....')
            self.device = torch.device('cuda:{}'.format(opt['train']['gpuid'][0]))
            self.gpuid = opt['train']['gpuid']
            self.net = net.to(self.device)
            self.logger.info('Loading Dual-Path-RNN parameters: {:.3f} Mb'.format(check_parameters(self.net)))
        else:
            self.logger.info('Load CPU ...........')
            self.device = torch.device('cpu')
            self.net = net.to(self.device)
            self.logger.info('Loading Dual-Path-RNN parameters: {:.3f} Mb'.format(check_parameters(self.net)))

        if opt['resume']['state']:
            best_path = os.path.join(opt['resume']['path'], 'best.pt')
            ckp = torch.load((best_path if os.path.exists(best_path) else os.path.join(opt['resume']['path'], 'last.pt')), map_location='cpu')
            self.cur_epoch = ckp['epoch']
            self.logger.info("Resume from checkpoint {}: epoch {:.3f}".format(opt['resume']['path'], self.cur_epoch))
            net.load_state_dict(ckp['model_state_dict'])
            self.net = net.to(self.device)
            optimizer.load_state_dict(ckp['optim_state_dict'])
            self.optimizer = optimizer
        else:
            self.net = net.to(self.device)
            self.optimizer = optimizer

        if opt['optim']['clip_norm']:
            self.clip_norm = opt['optim']['clip_norm']
            self.logger.info("Gradient clipping by {}, default L2".format(self.clip_norm))
        else:
            self.clip_norm = 0

    def train(self, epoch):
        self.logger.info('Start training from epoch: {:d}, iter: {:d}'.format(epoch, 0))
        self.net.train()
        total_sisnr_loss = 0.0
        total_accuracy = 0.0
        
        num_index = 1
        start_time = time.time()
        for egs in tqdm(self.train_dataloader):
            mix = egs['mix'].to(self.device)
            ref = egs['ref'].to(self.device)
            tar = egs['tar'].to(self.device)
            label = egs['label'].to(self.device)
            self.optimizer.zero_grad()
            print("step Training label: ", label)
            if self.gpuid:
                out_s, out_m, out_l, spk_pred = torch.nn.parallel.data_parallel(self.net, (mix, ref), device_ids=self.gpuid)
            else:
                out_s, out_m, out_l, spk_pred = self.net(mix, ref)
            sisnr_loss = (1 - self.alpha - self.beta) * SISNR_Loss(out_s, tar) + self.alpha * SISNR_Loss(out_l, tar) + self.beta * SISNR_Loss(out_l, tar)
            ce_loss = CE_Loss(spk_pred, label)
            accuracy = Accuracy(spk_pred, label)
            epoch_loss = sisnr_loss + self.gamma * ce_loss
            epoch_loss.backward()
            total_sisnr_loss += sisnr_loss.item()
            total_accuracy += accuracy.item()

            if self.clip_norm:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_norm)

            self.optimizer.step()
            if num_index % self.print_freq == 0:
                message = '<epoch:{:d}, iter:{:d}, lr:{:.3e}, sisnr loss:{:.3f}, accuracy: {:.3f}>'.format(
                    epoch, num_index, self.optimizer.param_groups[0]['lr'], total_sisnr_loss / num_index, total_accuracy / num_index)
                self.logger.info(message)
            num_index += 1
        end_time = time.time()
        total_sisnr_loss = total_sisnr_loss / num_index
        total_accuracy = total_accuracy / num_index
        message = 'Finished *** <epoch:{:d}, iter:{:d}, lr:{:.3e}, sisnr loss:{:.3f}, accuracy:{:.3f}, Total time:{:.3f} min> '.format(
            epoch, num_index, self.optimizer.param_groups[0]['lr'], total_sisnr_loss, total_accuracy, (end_time - start_time) / 60)
        self.logger.info(message)
        return total_sisnr_loss, total_accuracy

    def valid(self, epoch):
        self.logger.info('Start Validation from epoch: {:d}, iter: {:d}'.format(epoch, 0))
        self.net.eval()
        num_index = 1
        total_sisnr_loss = 0.0
        total_accuracy = 0.0
        start_time = time.time()
        with torch.no_grad():
            for egs in tqdm(self.valid_dataloader):
                mix = egs['mix'].to(self.device)
                ref = egs['ref'].to(self.device)
                tar = egs['tar'].to(self.device)
                label = egs['label'].to(self.device)
                self.optimizer.zero_grad()

                if self.gpuid:
                    out_s, out_m, out_l, spk_pred = torch.nn.parallel.data_parallel(self.net, (mix, ref), device_ids=self.gpuid)
                else:
                    out_s, out_m, out_l, spk_pred = self.net(mix, ref)
                sisnr_loss = (1 - self.alpha - self.beta) * SISNR_Loss(out_s, tar) + self.alpha * SISNR_Loss(out_l, tar) + self.beta * SISNR_Loss(out_l, tar)
                accuracy = Accuracy(spk_pred, label)
                total_sisnr_loss += sisnr_loss.item()
                total_accuracy += accuracy.item()
                if num_index % self.print_freq == 0:
                    message = '<epoch:{:d}, iter:{:d}, lr:{:.3e}, sisnr loss:{:.3f}, accuracy: {:.3f}>'.format(
                        epoch, num_index, self.optimizer.param_groups[0]['lr'], total_sisnr_loss / num_index, total_accuracy / num_index)
                    self.logger.info(message)
                num_index += 1
        end_time = time.time()
        total_sisnr_loss = total_sisnr_loss / num_index
        total_accuracy = total_accuracy / num_index
        message = 'Finished *** <epoch:{:d}, iter:{:d}, lr:{:.3e}, sisnr loss:{:.3f}, accuracy: {:.3f}, Total time:{:.3f} min> '.format(
            epoch, num_index, self.optimizer.param_groups[0]['lr'], total_sisnr_loss, total_accuracy, (end_time - start_time) / 60)
        self.logger.info(message)
        return total_sisnr_loss, total_accuracy

    def run(self):
        train_loss = []
        valid_loss = []
        train_accuracy = []
        valid_accuracy = []
        with torch.cuda.device(self.gpuid[0]):
            self.save_checkpoint(self.cur_epoch, best=False)
            v_sisnr_loss, v_accuracy = self.valid(self.cur_epoch)
            best_loss = v_sisnr_loss

            self.logger.info("Starting epoch from {:d}, loss = {:.4f}".format(self.cur_epoch, best_loss))
            no_improve = 0
            # start training part
            while self.cur_epoch < self.total_epoch:
                self.cur_epoch += 1
                t_sisnr_loss, t_accuracy = self.train(self.cur_epoch)
                v_sisnr_loss, v_accuracy = self.valid(self.cur_epoch)

                train_loss.append(t_sisnr_loss)
                valid_loss.append(v_sisnr_loss)
                train_accuracy.append(t_accuracy)
                valid_accuracy.append(v_accuracy)

                # schedule here
                self.scheduler.step(v_sisnr_loss + self.gamma * v_accuracy)

                if v_sisnr_loss >= best_loss:
                    no_improve += 1
                    self.logger.info('No improvement, Best Loss: {:.4f}, Accuracy: {:.4f}'.format(best_loss, v_accuracy))
                else:
                    best_loss = v_sisnr_loss
                    no_improve = 0
                    self.save_checkpoint(self.cur_epoch, best=True)
                    self.logger.info('Epoch: {:d}, Now Best Loss Change: {:.4f}, Accuracy: {:.4f}'.format(self.cur_epoch, best_loss, v_accuracy))

                if no_improve == self.early_stop:
                    self.logger.info("Stop training cause no improvement for {:d} epochs".format(no_improve))
                    break
            self.save_checkpoint(self.cur_epoch, best=False)
            self.logger.info("Training for {:d}/{:d} epochs done!".format(self.cur_epoch, self.total_epoch))

        # draw loss image
        plt.figure()
        plt.title('Loss of train and test')
        x = [i for i in range(self.cur_epoch)]
        plt.plot(x, train_loss, 'b-', label=u'train_loss', linewidth=0.8)
        plt.plot(x, valid_loss, 'c-', label=u'valid_loss', linewidth=0.8)
        plt.legend()
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig('loss.png')

        # draw accuracy
        plt.figure()
        plt.title('Accuracy of train and test')
        x = [i for i in range(self.cur_epoch)]
        plt.plot(x, train_accuracy, 'b-', label=u'train_accuracy', linewidth=0.8)
        plt.plot(x, valid_accuracy, 'c-', label=u'valid_accuracy', linewidth=0.8)
        plt.legend()
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.savefig('accuracy.png')

    def save_checkpoint(self, epoch, best=True):
        """
           save model
           best: the best model
        """
        os.makedirs(os.path.join(self.checkpoint, self.name), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optim_state_dict': self.optimizer.state_dict()
        }, os.path.join(self.checkpoint, self.name, '{0}.pt'.format('best' if best else 'last')))
