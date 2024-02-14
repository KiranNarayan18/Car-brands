import sys
from src.logger import logger, CustomException


from src.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.pipeline.model_training_pipeline import ModelTrainingPipeline

if __name__ == "__main__":
    
    try:

        # STAGE_NAME = 'Data Ingestion'
        # logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        # data_ingestion_obj = DataIngestionPipeline()
        # data_ingestion_obj.main()
        # logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")


        STAGE_NAME = 'Model Training'
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_training_obj = ModelTrainingPipeline()
        model_training_obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")




    except Exception as e:
        logger.error(CustomException(e, sys))
        