import os
import sys

from src.logger import logger, CustomException
from src.components.data_ingestion import DataIngestion
from src.config.configuration import ConfigurationManager


class DataIngestionPipeline:
    def __init__(self) -> None:
        pass


    def main(self):
        try:
            config = ConfigurationManager().DataIngestionConfig()
            data_ingestion = DataIngestion(config)
            data_ingestion.download_data()

        except Exception as error:
            logger.error(CustomException(error, sys))