import os
from pathlib import Path

from config.config import AppConfig


def build_triton_server(config: AppConfig, model):

    # Setup all paths
    server_path = Path(config.triton_path / "server" / "models")
    preprocess_path = Path(server_path / "preprocess")
    model_path = Path(server_path / config.model)
    ensemble_path = Path(server_path / "ensemble")
    config_template_path = Path(server_path / "config_template.pbtxt")

    # Move artifacts to server
    os.rename(model, Path(model_path / "1" / "model.onnx"))

    _write_ensemble_config(
        config_path=Path(ensemble_path / "config.pbtxt"), config=config
    )
    _write_model_config(
        config_path=Path(model_path / "config.pbtxt"),
        template=config_template_path,
        config=config,
    )
    _write_preprocess_config(
        config_path=Path(preprocess_path / "config.pbtxt"),
        template=config_template_path,
        config=config,
    )


def _write_ensemble_config(config_path: Path, config: AppConfig):
    tmp = "tmp.pbtxt"

    with open(config_path, "r") as file:
        data = file.readlines()

    with open(tmp, "w") as file:
        block = "none"

        for line in data:
            if "output [" in line:
                block = "out"
            elif "dims" in line:
                if block == "out":
                    line = "    dims: [ {0} ]\n".format(config.class_num)
            elif 'model_name: "preprocess"' in line:
                is_preprocess = True
            elif "model_name:" in line:
                if is_preprocess:
                    line = '      model_name: "{0}"\n'.format(config.model)
            file.write(line)

    os.rename(tmp, config_path)


def _write_model_config(config_path: Path, template: Path, config: AppConfig):
    name = config.model
    input_dim = "[ 1, 3, {0}, {1} ]".format(config.imsize[0], config.imsize[1])
    output_dim = "[ 1, {0} ]".format(config.class_num)
    _write_config(
        config_path=config_path,
        template=template,
        name=name,
        input_dim=input_dim,
        output_dim=output_dim,
    )


def _write_preprocess_config(config_path: Path, template: Path, config: AppConfig):
    name = "preprocess"
    input_dim = "[ -1 ]"
    output_dim = "[ 1, 3, {0}, {1} ]".format(config.imsize[0], config.imsize[1])
    _write_config(
        config_path=config_path,
        template=template,
        name=name,
        input_dim=input_dim,
        output_dim=output_dim,
    )


def _write_config(config_path: Path, template: Path, name, input_dim, output_dim):
    tmp = "tmp.pbtxt"

    if config_path.is_file():
        with open(config_path, "r") as file:
            data = file.readlines()
    else:
        with open(template, "r") as file:
            data = file.readlines()

    with open(tmp, "w") as file:
        block = "none"
        for line in data:
            if "name" in line and block == "none":
                line = 'name: "{0}"\n'.format(name)
            elif "input [" in line:
                block = "in"
            elif "output [" in line:
                block = "out"
            elif "dims" in line:
                if block == "in":
                    line = "    dims: {0}\n".format(input_dim)
                elif block == "out":
                    line = "    dims: {0}\n".format(output_dim)
            file.write(line)

    os.rename(tmp, config_path)


def main():
    config = AppConfig.parse_raw("./../config/config.yaml")
    model = Path("./../triton/server/models/resnet50/1/model.onnx")
    build_triton_server(config=config, model=model)


if __name__ == "__main__":
    main()
