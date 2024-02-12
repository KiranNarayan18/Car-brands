import os
import sys
from urllib import request

from src.logger import logger, CustomException
from src.entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self):
        try:
            os.makedirs(self.config.root_dir, exist_ok=True)
            self.filename = os.path.basename(self.config.url)
            request.urlretrieve(self.config.url, os.path.join(self.config.root_dir, self.filename))

        except Exception as error:
            logger.error(CustomException(error, sys))