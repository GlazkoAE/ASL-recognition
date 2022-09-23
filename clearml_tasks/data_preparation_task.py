import sys
from pathlib import Path

sys.path.append("./..")

from clearml import Dataset, Task, TaskTypes

from config.config import AppConfig
from src.data_preparation import get_class_num, main_actions


def main(config_path="../config/config.yaml"):
    config: AppConfig = AppConfig.parse_raw(filename=config_path)
    task: Task = Task.init(
        project_name=config.project_name,
        task_name="data preparation",
        task_type=TaskTypes.data_processing,
    )
    clearml_params = {"dataset_id": config.dataset_id}
    task.connect(clearml_params)
    task.execute_remotely()
    dataset_path = Dataset.get(**clearml_params).get_local_copy()
    config.dataset_path = Path(dataset_path)

    main_actions(config=config)
    class_num = get_class_num(config=config)

    dataset = Dataset.create(
        dataset_project=config.project_name, dataset_name=config.dataset_name
    )
    dataset.add_files(config.dataset_output_path)
    dataset.add_tags(str(config.imsize[0]) + "x" + str(config.imsize[1]))
    task.set_parameter("output_dataset_id", dataset.id)
    task.set_parameter("image_size", str(config.imsize))
    task.set_parameter("class_num", class_num)
    dataset.upload()
    dataset.finalize()


if __name__ == "__main__":
    main()
