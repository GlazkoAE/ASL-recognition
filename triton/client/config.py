from pathlib import Path
from typing import Union

from pydantic_yaml import YamlModel


class AppConfig(YamlModel):
    class_num: int
    grpc_endpoint: int
    server_host: str

    @classmethod
    def parse_raw(
        cls, filename: Union[str, Path] = "./config.yaml", *args, **kwargs
    ):
        with open(filename, "r") as f:
            data = f.read()
        return super().parse_raw(data, *args, **kwargs)
