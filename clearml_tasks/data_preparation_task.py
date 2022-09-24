import os
import sys
from pathlib import Path

from clearml import Dataset, Task, TaskTypes

from utils import load_dataset


def main(config_path="../config/config.yaml"):
    sys.path.append("./..")

    from config.config import AppConfig
    from src.data_preparation import get_class_num, prepare_data

    config: AppConfig = AppConfig.parse_raw(filename=config_path)
    task: Task = Task.init(
        project_name=config.project_name,
        task_name="data preparation",
        task_type=TaskTypes.data_processing,
    )

    clearml_params = {
        "dataset_name": config.dataset_name,
        "output_dataset_name": config.training_dataset_name,
        "dataset_id": "",
        "random_state": config.random_state,
    }
    task.connect(clearml_params)

    dataset_path, input_dataset_id = load_dataset(clearml_params)
    config.dataset_path = Path(dataset_path)
    config.dataset_output_path = os.path.join(config.dataset_path,
                                              "..",
                                              clearml_params["output_dataset_name"]
                                              )

    prepare_data(config=config)
    class_num = get_class_num(config=config)

    dataset = Dataset.create(
        dataset_project=config.project_name,
        dataset_name=clearml_params["output_dataset_name"],
    )
    dataset.add_files(config.dataset_output_path)
    dataset.add_tags(str(config.imsize[0]) + "x" + str(config.imsize[1]))

    task.set_parameter("dataset_id", input_dataset_id)
    task.set_parameter("output_dataset_id", dataset.id)
    task.set_parameter("image_size", str(config.imsize))
    task.set_parameter("class_num", class_num)
    task.set_parameter("spit_ratio", str(config.dataset_split_ratio))

    dataset.upload()
    dataset.finalize()


if __name__ == "__main__":
    main()
