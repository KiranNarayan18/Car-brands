import os
import sys
from glob import glob
import matplotlib.pyplot as plt
from glob import glob
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50

from src.utils import read_yaml_file
from src.logger import CustomException, logger
from src.config.configuration import ModelTrainingConfig

from pathlib import Path

class ModelTrainingPytorch:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.train_path = f"{self.config.root_dir}/Train"
        self.test_path = f"{self.config.root_dir}/Test"

        self.model_params = read_yaml_file(Path('params.yaml'))
        self.model_config = self.model_params.MODEL_CONFIG

    def train(self):
        try:
            
            folders = glob(f'{self.train_path}/*')

            num_classes = len(folders)

            print('num_classes', num_classes)  
            print("Training", self.train_path)
            print("Testing", self.test_path)
            print('self.model_config', self.model_config)

        except Exception as error:
            logger.error(CustomException(error, sys))


if __name__ == '__main__':
    obj = ModelTrainingPytorch()
    obj.train()
