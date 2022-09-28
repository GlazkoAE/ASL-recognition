import os
from pathlib import Path

from config.config import AppConfig


def build_triton_client(config: AppConfig, labels):

    # Setup all paths
    client_path = Path(config.triton_path / "client")
    client_config = Path(client_path / "config.yaml")
    _write_config(client_config=client_config, config=config)

    # Move artifacts to server
    os.rename(labels, Path(client_path / "labels.txt"))


def _write_config(client_config: Path, config: AppConfig):
    tmp = "tmp.yaml"

    with open(client_config, "r") as file:
        data = file.readlines()

    with open(tmp, "w") as file:
        for line in data:
            if "class_num" in line:
                line = "class_num: {0}\n".format(config.class_num)
            elif "server_host:" in line:
                line = "server_host: {0}\n".format(config.server_host)
            elif "grpc_endpoint:" in line:
                line = "grpc_endpoint: {0}\n".format(config.grpc_endpoint)
            file.write(line)

    os.rename(tmp, client_config)


def main():
    config = AppConfig.parse_raw("./../config/config.yaml")
    labels = Path("../triton/client/labels.txt")
    build_triton_server(config=config, labels=labels)


if __name__ == "__main__":
    main()
