import os
import sys

from src.logger import logger, CustomException
from src.utils import read_yaml
from src.constants import *
from src.entity import (
                            DataIngestionConfig
                        )

class ConfigurationManager:
    def __init__(
            self,
            config_file_path=CONFIG_FILE_PATH):
        
        
        self.config = read_yaml(config_file_path)
        

    def DataIngestionConfig(self):
        try:
            config = self.config.data_ingestion

            os.makedirs(config.root_dir, exist_ok=True)

            data_ingestion_config = DataIngestionConfig(
                root_dir = config.root_dir,
                source_url = config.source_url
            )

            return data_ingestion_config

        except Exception as error:
            logger.error(error)
        

