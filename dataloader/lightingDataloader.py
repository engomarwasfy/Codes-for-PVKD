from typing import Any

import pytorch_lightning as pl
from builder import data_builder
from config.config import load_config_data
from dataloader.pc_dataset import get_SemKITTI_label_name
import numpy as np

class lightingDataloader(pl.LightningDataModule):
    def __init__(self, config_path):
        super().__init__()
        self.config_path = config_path
        self.configs = load_config_data(self.config_path)
        self.dataset_config = self.configs['dataset_params']
        self.train_dataloader_config = self.configs['train_data_loader']
        self.val_dataloader_config = self.configs['val_data_loader']
        self.model_config = self.configs['model_params']
        self.grid_size = self.model_config['output_shape']
        self.val_batch_size = self.val_dataloader_config['batch_size']
        self.train_batch_size = self.train_dataloader_config['batch_size']
        self.SemKITTI_label_name = get_SemKITTI_label_name(self.dataset_config["label_mapping"])
        self.unique_label = np.asarray(sorted(list(self.SemKITTI_label_name.keys())))[1:] - 1
        self.unique_label_str = [self.SemKITTI_label_name[x] for x in self.unique_label]
        self.setup(None)

    def setup(self,stage):
        self.train_dataset_loader, self.val_dataset_loader = data_builder.build(self.dataset_config,
                                                                                self.train_dataloader_config,
                                                                                self.val_dataloader_config,
                                                                                grid_size=self.grid_size)
        #self.train_dataset_loader = setup_dataloaders(self.train_dataset_loader)  # Scale your dataloaders
        #self.val_dataset_loader = setup_dataloaders(self.val_dataset_loader)

    def train_dataloader(self):
        return self.train_dataset_loader

    def val_dataloader(self):
        return self.val_dataset_loader

    def test_dataloader(self):
        return self.val_dataset_loader

    def predict_dataloader(self):
        return self.val_dataset_loader

    def state_dict(self) -> dict[str, Any]:
        """Called when saving a checkpoint, implement to generate and save datamodule state.

        Returns:
            A dictionary containing datamodule state.
        """
        return dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Called when loading a checkpoint, implement to reload datamodule state given datamodule state_dict.

        Args:
            state_dict: the datamodule state returned by ``state_dict``.
        """
        pass