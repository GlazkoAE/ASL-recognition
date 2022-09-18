from typing import Union
from pathlib import Path
from pydantic_yaml import YamlModel


class AppConfig(YamlModel):
    # common
    random_state: int

    # data
    dataset_path: Path
    dataset_output_path: Path
    training_dataset_path: Path

    # training
    epochs: int
    batch_size: int
    lr: float

    @classmethod
    def parse_raw(cls, filename: Union[str, Path] = "config.yaml",
                  *args, **kwargs):
        with open(filename, 'r') as f:
            data = f.read()
        return super().parse_raw(data, *args, **kwargs)
