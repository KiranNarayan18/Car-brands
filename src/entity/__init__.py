from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionConfig:
    url: str
    root_dir: Path

@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    model_dir: Path