from pathlib import Path
from typing import Union

from pydantic_yaml import YamlModel


class AppConfig(YamlModel):
    # common
    random_state: int

    # clearml
    project_name: str
    dataset_name: str
    output_dataset_name: str
    # dataset_id: str

    # data
    dataset_path: Path
    dataset_output_path: Path
    training_dataset_path: Path
    imsize: tuple
    dataset_split_ratio: tuple
    # class_num: int

    # training
    model: str
    epochs: int
    batch_size: int
    lr: float
    num_workers: int

    @classmethod
    def parse_raw(
        cls, filename: Union[str, Path] = "config/config.yaml", *args, **kwargs
    ):
        with open(filename, "r") as f:
            data = f.read()
        return super().parse_raw(data, *args, **kwargs)
