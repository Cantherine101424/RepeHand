import os
import os.path as osp
import math
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms

from common.timer import Timer
from common.logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
from main.config import cfg
import re
from main.model import get_tinyvit_model, get_resnet_model, get_rcvit_model
from data.dataset import MultipleDatasets
import numpy as np
import random

# dynamic dataset import
for i in range(len(cfg.trainset_3d)):
    exec('from ' + cfg.trainset_3d[i] + ' import ' + cfg.trainset_3d[i])
for i in range(len(cfg.trainset_2d)):
    exec('from ' + cfg.trainset_2d[i] + ' import ' + cfg.trainset_2d[i])
exec('from ' + cfg.testset + ' import ' + cfg.testset)


class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return

class Trainer(Base):
    def __init__(self):
        super(Trainer, self).__init__(log_name='train_logs.txt')
        self.best_val_loss = float('inf')
        self.scheduler = None
        self.model = None
        self.optimizer = None
        self.start_epoch = 0
        self.checkpoint = None
        self.latest_checkpoint = None

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_optimizer(self, model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        return optimizer

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating train dataset...")
        trainset3d_loader = []
        for i in range(len(cfg.trainset_3d)):
            trainset3d_loader.append(eval(cfg.trainset_3d[i])(transforms.ToTensor(), "train"))
        trainset2d_loader = []
        for i in range(len(cfg.trainset_2d)):
            trainset2d_loader.append(eval(cfg.trainset_2d[i])(transforms.ToTensor(), "train"))

        valid_loader_num = 0
        if len(trainset3d_loader) > 0:
            trainset3d_loader = [MultipleDatasets(trainset3d_loader, make_same_len=False)]
            valid_loader_num += 1
        else:
            trainset3d_loader = []

        if len(trainset2d_loader) > 0:
            trainset2d_loader = [MultipleDatasets(trainset2d_loader, make_same_len=False)]
            valid_loader_num += 1
        else:
            trainset2d_loader = []

        if valid_loader_num > 1:
            trainset_loader = MultipleDatasets(trainset3d_loader + trainset2d_loader, make_same_len=True)
        else:
            trainset_loader = MultipleDatasets(trainset3d_loader + trainset2d_loader, make_same_len=False)

        self.itr_per_epoch = math.ceil(len(trainset_loader) / cfg.num_gpus / cfg.train_batch_size)
        print('itr_per_epoch:', self.itr_per_epoch)
        self.batch_generator = DataLoader(dataset=trainset_loader, batch_size=cfg.num_gpus * cfg.train_batch_size,
                                          shuffle=True, num_workers=cfg.num_thread, pin_memory=True, drop_last=True)

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_tinyvit_model('train')
        model = DataParallel(model).cuda()
        self.optimizer = self.get_optimizer(model)
        if cfg.continue_train_kd:
            self.start_epoch, model, self.optimizer, self.best_val_loss = self.load_model_kd(model, self.optimizer)
        elif cfg.continue_train:
            self.start_epoch, model, self.optimizer, self.best_val_loss, self.checkpoint, self.latest_checkpoint = self.load_model(model, self.optimizer)
        else:
            self.start_epoch = 0
            self.best_val_loss = float('inf')

        model.train()
        self.model = model


    def load_model(self, model, optimizer):
        try:
            model_file_list = glob.glob(osp.join(cfg.model_dir, '*.pth.tar'))
            if not model_file_list:
                self.logger.warning('No checkpoint found, starting from scratch')
                return 0, model, optimizer, float('inf')

            latest_checkpoint = max([f for f in model_file_list if 'snapshot_' in f],
                                    key=lambda x: int(re.search(r'snapshot_(\d+)', x).group(1)))

            self.logger.info(f'Loading checkpoint from {latest_checkpoint}')
            checkpoint = torch.load(latest_checkpoint, map_location='cpu')

            model.load_state_dict(checkpoint['network'])
            self.logger.info('Model state restored')

            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                self.logger.info('Optimizer state restored')

            if 'torch_rng_state' in checkpoint:
                torch.set_rng_state(checkpoint['torch_rng_state'])
                np.random.set_state(checkpoint['numpy_rng_state'])
                random.setstate(checkpoint['random_rng_state'])
                if checkpoint['dataloader_seed'] is not None and hasattr(self, 'batch_generator') and \
                        hasattr(self.batch_generator, 'generator'):
                    self.batch_generator.generator.set_state(checkpoint['dataloader_seed'])
                self.logger.info('Random states restored (PyTorch/Numpy/Random/DataLoader)')

            start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))

            if 'lr' in checkpoint:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = checkpoint['lr']
                self.logger.info(f'Learning rate restored to {checkpoint["lr"]}')

            self.logger.info(f'Successfully loaded checkpoint from epoch {checkpoint["epoch"]}')
            self.logger.info(f'Training will resume from epoch {start_epoch}')
            self.logger.info(f'Best validation loss so far: {best_val_loss}')

            return start_epoch, model, optimizer, best_val_loss, checkpoint, latest_checkpoint

        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            self.logger.warning('Starting training from scratch')
            return 0, model, optimizer, float('inf')

    def _make_model_t(self):
        # prepare network
        self.logger.info("Creating teacher graph and optimizer...")
        model_t = get_resnet_model('test')
        model_t = DataParallel(model_t).cuda()
        # optimizer = self.get_optimizer(model)
        model_t = self.load_model_t(model_t)
        model_t.eval()
        #
        # self.start_epoch = start_epoch
        self.model_t = model_t
        # self.optimizer = optimizer

    def save_best_model(self, state, is_best=False):
        state.update({
            'torch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'dataloader_seed': self.batch_generator.generator.get_state() if self.batch_generator.generator else None
        })
        if is_best:
            best_file_path = osp.join(cfg.model_dir, 'best_model.pth.tar')
            torch.save(state, best_file_path)
            self.logger.info("Write best snapshot into {}".format(best_file_path))
        else:
            self.logger.info("save_best_model is wrong")

    def save_model(self, state, epoch, is_best=False):
        state.update({
            'torch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'dataloader_seed': self.batch_generator.generator.get_state() if self.batch_generator.generator else None
        })
        file_path = osp.join(cfg.model_dir, f'snapshot_{epoch}.pth.tar')
        torch.save(state, file_path)
        self.logger.info(f"Saved snapshot to {file_path}")


    def load_model_kd(self, model, optimizer):
        model_path = osp.join('./best_model.pth.tar')
        self.logger.info(f'Loading checkpoint from {model_path}')
        checkpoint = torch.load(model_path)

        model.load_state_dict(checkpoint['network'])
        self.logger.info('Model state restored')

        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        self.logger.info(f'Successfully loaded checkpoint from epoch {checkpoint["epoch"]}')
        self.logger.info(f'Training will resume from epoch {start_epoch}')
        self.logger.info(f'Best validation loss so far: {best_val_loss}')

        return start_epoch, model, optimizer, best_val_loss

    def load_model_t(self, model):
        ckpt_path = osp.join('./snapshot_29.pth.tar')
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['network'], strict=False)
        self.logger.info('Load teacher checkpoint from {}'.format(ckpt_path))
        return model

class Tester(Base):
    def __init__(self, test_epoch):
        self.test_epoch = test_epoch
        super(Tester, self).__init__(log_name='test_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        testset_loader = eval(cfg.testset)(transforms.ToTensor(), "test")
        batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.num_gpus * cfg.test_batch_size,
                                     shuffle=False, num_workers=cfg.num_thread, pin_memory=True)

        self.testset = testset_loader
        self.batch_generator = batch_generator

    def _make_model(self):
        if self.test_epoch == 'best_model':
            model_path = os.path.join(cfg.model_dir, 'best_model.pth.tar')
        else:
            model_path = os.path.join(cfg.model_dir, 'snapshot_{}.pth.tar'.format(self.test_epoch))

        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))

        # prepare network
        model = get_tinyvit_model('test')
        model = DataParallel(model).cuda()
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'])
        model.eval()
        self.model = model

    def _evaluate(self, preds, cur_sample_idx):
        eval_result=self.testset.evaluate(preds, cur_sample_idx)
        return eval_result

    def _sub_evaluate(self, preds):
        value = self.testset.evaluate(preds)
        return value

    def _print_eval_result(self, eval_result):
        eval_result=self.testset.print_eval_result(eval_result)
        return eval_result


