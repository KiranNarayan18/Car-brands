import os
import sys

from src.logger import logger, CustomException
from src.utils import read_yaml_file
from src.constants import *
from src.entity import (
                            DataIngestionConfig,
                            ModelTrainingConfig
                        )

class ConfigurationManager:
    def __init__(
            self,
            config_file_path=CONFIG_FILE_PATH):
        

        self.config = read_yaml_file(config_file_path)
        

    def DataIngestionConfig(self):
        try:
            config = self.config.data_ingestion

            os.makedirs(config.root_dir, exist_ok=True)

            data_ingestion_config = DataIngestionConfig(
                root_dir = config.root_dir,
                url = config.url
            )

            return data_ingestion_config

        except Exception as error:
            logger.error(CustomException(error, sys))
        

    def ModelTraining(self):

        try:
            config = self.config.model_training

            os.makedirs(config.model_dir, exist_ok=True)

            model_training = ModelTrainingConfig(
                root_dir=config.root_dir,
                model_dir=config.model_dir
            )

            return model_training
        
        except Exception as error:
            logger.error(CustomException(error, sys))