import sys
from pathlib import Path

from clearml import Dataset, Task, TaskTypes


def main(config_path="../config/config.yaml"):
    sys.path.append("./..")

    from config.config import AppConfig
    from src.data_validation import validate_data

    config: AppConfig = AppConfig.parse_raw(filename=config_path)

    task: Task = Task.init(
        project_name=config.project_name,
        task_name="data validation",
        task_type=TaskTypes.data_processing,
    )
    clearml_params = {"dataset_name": config.dataset_name,
                      "dataset_id": ""}
    task.connect(clearml_params)
    # task.execute_remotely()
    if clearml_params["dataset_id"]:
        dataset_path = Dataset.get(dataset_id=clearml_params["dataset_id"],
                                   ).get_local_copy()
    else:
        dataset_path = Dataset.get(dataset_name=clearml_params["dataset_name"],
                                   dataset_tags=["latest"]
                                   ).get_local_copy()

    config.dataset_path = Path(dataset_path)
    validate_data(config=config)


if __name__ == "__main__":
    main()
