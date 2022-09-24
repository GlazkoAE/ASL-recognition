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

    clearml_params = config.dict()
    clearml_params["dataset_id"] = ""
    task.connect(clearml_params)

    if clearml_params["dataset_id"]:
        dataset = Dataset.get(dataset_id=clearml_params["dataset_id"])
        dataset_path = dataset.get_local_copy()
    else:
        dataset = Dataset.get(dataset_name=clearml_params["training_dataset_name"])
        dataset_path = dataset.get_local_copy()
        task.set_parameter("dataset_id", dataset.id)

    config.training_dataset_path = Path(dataset_path)
    config.class_num = clearml_params["class_num"]
    model_path = train_model(config=config)

    task.upload_artifact(name='onnx_model', artifact_object=model_path)
    os.remove(model_path)


if __name__ == "__main__":
    main()
