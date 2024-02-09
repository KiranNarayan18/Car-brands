import sys
from src.logger import logger, CustomException


if __name__ == "__main__":
    
    try:
        a = 1/0
        logger.info("Starting")
    except Exception as e:
        logger.error(CustomException(e, sys))
        