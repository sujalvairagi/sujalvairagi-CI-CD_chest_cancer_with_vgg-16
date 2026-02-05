from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    split_dir: Path
    train_dir: Path
    val_dir: Path
    test_dir: Path
    split_ratio: dict



@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int




@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list
    params_warmup_epochs: int
    params_fine_tune_epochs: int
    params_fine_tune_layers: int
    params_warmup_lr: float
    params_fine_tune_lr: float
    params_weights: str



@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: dict
    params_image_size: list
    params_batch_size: int
    mlflow_uri: str


@dataclass(frozen=True)
class CTGateConfig:
    root_dir: Path
    data_dir: Path
    model_path: Path
    params_image_size: list
    params_batch_size: int
    params_epochs: int
    params_learning_rate: float
    mlflow_uri: str



