import sys

from src.logger import logger, CustomException

from src.config.configuration import ConfigurationManager
from src.components.model_training import ModelTraining


class ModelTrainingPipeline:
    def __init__(self) -> None:
        pass


    def main(self):
        try:
            config = ConfigurationManager()
            config = config.ModelTraining()

            training_obj = ModelTraining(config)
            training_obj.train()


        except Exception as error:
            logger.error(CustomException(error, sys))