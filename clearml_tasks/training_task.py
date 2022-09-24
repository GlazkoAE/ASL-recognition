import os
import sys
from pathlib import Path

sys.path.append("./..")


def main(config_path="../config/config.yaml"):
    from clearml import Dataset, Task, TaskTypes

    from config.config import AppConfig
    from src.training import train_model

    config: AppConfig = AppConfig.parse_raw(filename=config_path)

    task: Task = Task.init(
        project_name=config.project_name,
        task_name="training",
        task_type=TaskTypes.training,
    )

    clearml_params = {
        "dataset_id": "",
        "dataset_name": config.training_dataset_name,
        "project_name": config.project_name,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "log_every": config.log_every,
        "lr": config.lr,
        "model": config.model,
        "num_workers": config.num_workers,
        "random_state": config.random_state,
        "class_num": config.class_num,
        "split_ratio": config.dataset_split_ratio,
        "image_size": config.imsize,
    }
    task.connect(clearml_params)

    if clearml_params["dataset_id"]:
        dataset = Dataset.get(dataset_id=clearml_params["dataset_id"])
        dataset_path = dataset.get_local_copy()
    else:
        dataset = Dataset.get(dataset_name=clearml_params["dataset_name"])
        dataset_path = dataset.get_local_copy()
        task.set_parameter("dataset_id", dataset.id)

    config.training_dataset_path = Path(dataset_path)
    config.class_num = clearml_params["class_num"]
    model_path, labels_map = train_model(config=config)

    task.upload_artifact(name="onnx_model", artifact_object=model_path)
    task.upload_artifact(name="labels_map", artifact_object=labels_map)
    os.remove(model_path)


if __name__ == "__main__":
    main()
