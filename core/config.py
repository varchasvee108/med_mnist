from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass
class ModelConfig:
    img_size: tuple[int, int]
    patch_size: tuple[int, int]
    in_channels: int
    embd_dim: int
    num_heads: int
    num_layers: int
    mlp_ratio: int
    dropout: float
    num_classes: int


@dataclass
class TrainingConfig:
    max_steps: int
    warmup_steps: int
    weight_decay: float
    betas: tuple[float, float]
    lr: float
    batch_size: int


@dataclass
class DataConfig:
    dataset: str
    data_dir: str
    num_workers: int


@dataclass
class LoggingConfig:
    log_interval: int
    save_dir: str


@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    logging: LoggingConfig

    @classmethod
    def load_config(cls, path: str) -> "Config":
        config_path = Path(path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        if "model" not in data or "training" not in data:
            raise ValueError("Invalid config structure")

        training_data = data["training"].copy()
        training_data["betas"] = tuple(training_data["betas"])

        model_data = data["model"].copy()
        model_data["patch_size"] = tuple(model_data["patch_size"])
        model_data["img_size"] = tuple(model_data["img_size"])

        return cls(
            model=ModelConfig(**model_data),
            training=TrainingConfig(**training_data),
            data=DataConfig(**data["data"]),
            logging=LoggingConfig(**data["logging"]),
        )
