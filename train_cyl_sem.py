# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py


import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, StochasticWeightAveraging, ModelCheckpoint
from pytorch_lightning.cli import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm

from dataloader.lightingDataloader import lightingDataloader
from utils.metric_util import per_class_iu, fast_hist_crop, IoU
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor


from utils.load_save_util import load_checkpoint
from pytorch_lightning import *
import torch.nn.functional as F
import warnings
import pytorch_lightning as pl

from utils.schedulers import cosine_schedule_with_warmup

warnings.filterwarnings("ignore")
class Lite(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.config_path = args.config_path
        self.configs = load_config_data(self.config_path)
        self.dataset_config = self.configs['dataset_params']
        self.model_config = self.configs['model_params']
        self.train_hypers = self.configs['train_params']
        self.grid_size = self.model_config['output_shape']
        self.num_class = self.model_config['num_class']
        self.ignore_label = self.dataset_config['ignore_label']
        self.model_load_path = self.train_hypers['model_load_path']
        self.model_save_path = self.train_hypers['model_save_path']
        self.my_model = model_builder.build(self.model_config)
        self.SemKITTI_label_name = get_SemKITTI_label_name(self.dataset_config["label_mapping"])
        self.unique_label = np.asarray(sorted(list(self.SemKITTI_label_name.keys())))[1:] - 1
        self.unique_label_str = [self.SemKITTI_label_name[x] for x in self.unique_label]
        self.hist_list = []
        self.val_loss_list = []
        self.best_val_miou=0
        self.loss_list=[]
        self.val_iou = IoU(self.configs['dataset_params'], compute_on_step=True)
        if os.path.exists(self.model_load_path) and self.configs['train_params']['manual_load_ckpt'] and not self.configs['train_params']['load_ckpt'] :
            print("manual Model load")
            self.my_model = load_checkpoint(self.model_load_path, self.my_model)

    def training_step(self, batch  ,batch_idx):
        _, train_vox_label, train_grid, _, train_pt_fea = batch
        train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor) for i in  train_pt_fea]
        # train_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in train_grid]
        train_vox_ten = [torch.from_numpy(i).type(torch.FloatTensor) for i in train_grid]
        point_label_tensor = train_vox_label.type(torch.LongTensor).type_as(train_vox_label)
        # forward + backward + optimize
        outputs = self.my_model(train_pt_fea_ten, train_vox_ten, point_label_tensor.shape[0])  # train_batch_size)
        loss=0
        lovasz_loss=0
        ordinary_loss=0
        additonal_lovasz_loss=0
        additonal_ordinary_loss=0
        for output in outputs:

            additonal_lovasz_loss=self.lovasz_softmax(torch.nn.functional.softmax(output), point_label_tensor, ignore=0)
            additonal_ordinary_loss=self.loss_func(output, point_label_tensor)
            lovasz_loss+=additonal_lovasz_loss
            ordinary_loss+=additonal_ordinary_loss
            loss += additonal_lovasz_loss+additonal_ordinary_loss
        self.loss_list.append(loss.item())
        self.log("train/total_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,enable_graph=True)
        self.log("train/lovasz_loss", lovasz_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,enable_graph=True)
        self.log("train/ordinary_loss", ordinary_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,enable_graph=True)
        del outputs
        del train_pt_fea_ten,train_vox_ten ,point_label_tensor
        return loss
    def validation_step(self,batch,batch_idx):
        _, val_vox_label, val_grid, val_pt_labs, val_pt_fea = batch

        val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor) for i in
                          val_pt_fea]
        val_grid_ten = [torch.from_numpy(i).type(torch.FloatTensor) for i in val_grid]
        val_label_tensor = val_vox_label.type(torch.LongTensor)

        predict_labels = self.my_model(val_pt_fea_ten, val_grid_ten, val_label_tensor.shape[0])  # val_batch_size)
        cur_dev = predict_labels[0].get_device()
        if (cur_dev < 0):
            cur_dev = 0
        # aux_loss = loss_fun(aux_outputs, point_label_tensor)
        lovasz_loss = 0
        ordinary_loss = 0
        lovasz_loss =self.lovasz_softmax(torch.nn.functional.softmax(predict_labels[0]).to(cur_dev), val_label_tensor.to(cur_dev),ignore=0)
        ordinary_loss = self.loss_func(predict_labels[0].to(cur_dev), val_label_tensor.to(cur_dev))
        loss = lovasz_loss + ordinary_loss
        #removed ignore 0
        self.val_loss_list.append(loss)

        predict_labels_ = torch.argmax(predict_labels[0], dim=1)
        for count, i_val_grid in enumerate(val_grid):
            self.hist_list.append(fast_hist_crop(predict_labels_[
                                                count, val_grid[count][:, 0], val_grid[count][:, 1],
                                                val_grid[count][:, 2]], val_pt_labs[count],
                                            self.unique_label))
        
        iou = per_class_iu(sum(self.hist_list))
        print('Validation per class iou: ')
        for class_name, class_iou in zip(self.unique_label_str, iou):
            print('%s : %.2f%%' % (class_name, class_iou * 100))
        val_miou = np.nanmean(iou) * 100
        
        del val_vox_label, val_grid, val_pt_fea, val_grid_ten

        # save model if performance is improved
        if self.best_val_miou < val_miou:
            self.best_val_miou = val_miou
            torch.save(self.my_model.state_dict(), self.model_save_path)
        print('Current val miou is %.3f while the best val miou is %.3f' %
              (val_miou, self.best_val_miou))
        #print('Current val loss is %.3f' %
         #     (np.mean(self.val_loss_list)))
        self.log("val/lovasz_loss", lovasz_loss, on_step = True, on_epoch = True, prog_bar = True, logger = True,enable_graph=True)
        self.log("val/ordinary_loss", ordinary_loss, on_step = True, on_epoch = True, prog_bar = True, logger = True,enable_graph=True)
        self.log("val/total_loss", loss, on_step = True, on_epoch = True, prog_bar = True, logger = True,enable_graph=True)
        self.log("val/mIoU", val_miou, on_step = True, on_epoch = True, prog_bar = True, logger = True,enable_graph=True)
        self.log("val/best_miou", self.best_val_miou, on_step = True, on_epoch = True, prog_bar = True, logger = True,enable_graph=True)
        del predict_labels
        return loss
    def validation_epoch_end(self, outputs):
        iou, best_miou = self.val_iou.compute()
        mIoU = np.nanmean(iou)
        str_print = ''
        self.log('val/mIoU', mIoU, on_epoch = True, prog_bar = True, logger = True,enable_graph=True)
        self.log('val/best_miou', best_miou, on_epoch = True, prog_bar = True, logger = True,enable_graph=True)
        str_print += 'Validation per class iou: '
        try:
            for class_name, class_iou in zip(self.val_iou.unique_label_str, iou):
                str_print += '\n%s : %.2f%%' % (class_name, class_iou * 100)

            str_print += '\nCurrent val miou is %.3f while the best val miou is %.3f' % (mIoU * 100, best_miou * 100)
            self.print(str_print)
        except:
            print('Error in printing iou')

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.my_model.parameters(), lr=self.train_hypers["learning_rate"])
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        self.loss_func, self.lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                       num_class=self.num_class, ignore_label=self.ignore_label)
        if self.configs['train_params']['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.args['train_params']["learning_rate"])
        elif self.configs['train_params']['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.configs['train_params']["learning_rate"],
                                        momentum=self.configs['train_params']["momentum"],
                                        weight_decay=self.configs['train_params']["weight_decay"],
                                        nesterov=self.configs['train_params']["nesterov"])
        else:
            raise NotImplementedError

        if self.configs['train_params']["lr_scheduler"] == 'StepLR':
            lr_scheduler = StepLR(
                optimizer,
                step_size=self.configs['train_params']["decay_step"],
                gamma=self.configs['train_params']["decay_rate"]
            )
        elif self.configs['train_params']["lr_scheduler"] == 'ReduceLROnPlateau':
            lr_scheduler = ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=self.configs['train_params']["decay_rate"],
                patience=self.configs['train_params']["decay_step"],
                verbose=True
            )
        elif self.configs['train_params']["lr_scheduler"] == 'CosineAnnealingLR':
            lr_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.configs['train_params']['max_num_epochs'] - 4,
                eta_min=1e-5,
            )
        elif self.configs['train_params']["lr_scheduler"] == 'CosineAnnealingWarmRestarts':
            from functools import partial
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=partial(
                    cosine_schedule_with_warmup,
                    num_epochs=self.configs['train_params']['max_num_epochs'],
                    batch_size=self.configs['train_data_loader']['batch_size'],
                    dataset_size=self.configs['dataset_params']['training_size'],
                    num_gpu=1
                ),
            )
        else:
            raise NotImplementedError

        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step' if self.configs['train_params'][
                                      "lr_scheduler"] == 'CosineAnnealingWarmRestarts' else 'epoch',
            'frequency': 1
        }

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val/mIoU'
        }

       # return [optimizer], [lr_scheduler]

    def on_after_backward(self) -> None:
        """
        Skipping updates in case of unstable gradients
        https://github.com/Lightning-AI/lightning/issues/4956
        """
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break
        if not valid_gradients:
            print(f'detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()
def main(args):
   # Lite(  accelerator="cuda" ).run(args)
   config_path = args.config_path
   configs = load_config_data(config_path)
   tb_logger = pl_loggers.TensorBoardLogger(os.getcwd(), name=configs['train_params']['log_dir_name'], default_hp_metric=True, log_graph=True)
   os.makedirs(os.getcwd()+'/pl_logs', exist_ok=True)
   #profiler = SimpleProfiler(filename=os.getcwd()+configs['train_params']['log_dir_name']+'/profiler.txt')
   checkpoint_callback = ModelCheckpoint(
       monitor=configs['train_params']['monitor'],
       mode='max',
       save_last=True,
       save_top_k=configs['train_params']['save_top_k'],
       verbose=True,
       save_on_train_epoch_end=True,
       dirpath=configs['train_params']['save_dir_path'],
       #auto_insert_metric_name=True,
       every_n_train_steps=configs['train_params']['checkpoint_every_n_steps'],
   )
   swa = []
   if configs['train_params']['swa_enabled']:
    swa = [StochasticWeightAveraging(swa_epoch_start=configs['train_params']['swa_start'] ,
                                     annealing_epochs=configs['train_params']['swa_annealing_epochs'],
                                     swa_lrs=configs['train_params']['swa_lr'])]

   # reproducibility
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = True
       # init trainer
   print('Start training...')
   trainer = pl.Trainer(accelerator='cuda',
                            max_epochs=configs['train_params']['max_num_epochs'],
                            log_every_n_steps=configs['train_params']['log_every_n_steps'],
                            enable_checkpointing=True,
                            auto_lr_find=True,
                            auto_scale_batch_size=True,
                            benchmark=True,
                            check_val_every_n_epoch=1,
                            gradient_clip_val=configs['train_params']['gradient_clip_val'],
                            accumulate_grad_batches=configs['train_params']['accumulate_grad_batches'],
                            precision=32,
                            num_sanity_val_steps=configs['train_params']['num_sanity_val_steps'],
                            detect_anomaly=False,
                            callbacks=[checkpoint_callback,
                                   LearningRateMonitor(logging_interval='step',log_momentum=True),
                                   EarlyStopping(monitor=configs['train_params']['monitor'],
                                                 patience=configs['train_params']['patience'],
                                                 mode='max',
                                                 verbose=True),
                                   ] + swa,
                        logger=tb_logger,
                        #profiler=profiler,
                            enable_progress_bar=True,
                            )

   train_loader =lightingDataloader(args.config_path)

   model = Lite()
   if(not configs['train_params']['load_ckpt']):
    trainer.fit(model, train_dataloaders=train_loader.train_dataloader(),
               val_dataloaders=train_loader.val_dataloader())
   else:
       trainer.fit(model, train_dataloaders=train_loader.train_dataloader(),
                   val_dataloaders=train_loader.val_dataloader(),
                   ckpt_path=configs['train_params']['ckpt_path'])





if __name__ == '__main__':
    # Training settings

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti.yaml')
    args = parser.parse_args()
    config_path = args.config_path
    configs = load_config_data(config_path)
    print(' '.join(sys.argv))
    print(args)
    main(args)
