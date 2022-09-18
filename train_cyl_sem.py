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
from tqdm import tqdm

from dataloader.lightingDataloader import lightingDataloader
from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint
from pytorch_lightning import *
import torch.nn.functional as F
import warnings
import pytorch_lightning as pl

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
        self.loss_list=[]
        if os.path.exists(self.model_load_path):
            self.my_model = load_checkpoint(self.model_load_path, self.my_model)
    def training_step(self, batch  ,batch_idx):
        _, train_vox_label, train_grid, _, train_pt_fea = batch
        train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor) for i in  train_pt_fea]
        # train_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in train_grid]
        train_vox_ten = [torch.from_numpy(i).type(torch.FloatTensor) for i in train_grid]
        point_label_tensor = train_vox_label.type(torch.LongTensor).type_as(train_vox_label)
        # forward + backward + optimize
        outputs = self.my_model(train_pt_fea_ten, train_vox_ten, point_label_tensor.shape[0])  # train_batch_size)
        loss = self.lovasz_softmax(torch.nn.functional.softmax(outputs), point_label_tensor, ignore=0) + self.loss_func(
            outputs, point_label_tensor)
        self.loss_list.append(loss.item())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
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
        loss = self.lovasz_softmax(torch.nn.functional.softmax(predict_labels).to(cur_dev), val_label_tensor.to(cur_dev),ignore=0) \
               + self.loss_func(predict_labels.to(cur_dev), val_label_tensor.to(cur_dev))\
               + self.loss_func(predict_labels.to(cur_dev), val_label_tensor.to(cur_dev))
        #removed ignore 0
        self.val_loss_list.append(loss)

        '''
        predict_labels = torch.argmax(predict_labels, dim=1)
        predict_labels = predict_labels.detach()
        for count, i_val_grid in enumerate(val_grid):
            self.hist_list.append(fast_hist_crop(predict_labels[
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
        print('Current val loss is %.3f' %
              (np.mean(self.val_loss_list)))
        '''
        self.log("val_loss", loss, on_step = True, on_epoch = True, prog_bar = True, logger = True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.my_model.parameters(), lr=self.train_hypers["learning_rate"])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        self.loss_func, self.lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                       num_class=self.num_class, ignore_label=self.ignore_label)
        return [optimizer], [lr_scheduler]
def main(args):
   # Lite(  accelerator="cuda" ).run(args)
   model = Lite()
   data_loader =lightingDataloader(args.config_path)
   trainer = pl.Trainer(max_epochs=40, accelerator="cuda")
   trainer.fit(model, train_dataloaders=data_loader.train_dataloader(), val_dataloaders=data_loader.val_dataloader())


if __name__ == '__main__':
    # Training settings

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
