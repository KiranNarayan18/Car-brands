import os
import sys
import yaml
from box import ConfigBox
from pathlib import Path
from ensure import ensure_annotations

from src.logger import logger, CustomException

@ensure_annotations
def read_yaml_file(file_path: Path):
    try:
        with open(file_path, "r") as f:
            content = yaml.safe_load(f)

            logger.info(f"yaml file: {file_path} loaded successfully")
            return ConfigBox(content)            
        
    except Exception as error:
        logger.error(CustomException(error, sys))        
        raise error