import sys
from pathlib import Path

sys.path.append("./..")

from clearml import Dataset, Task, TaskTypes

from config.config import AppConfig
from src.data_validation import main_actions


def main(config_path="../config/config.yaml"):
    task: Task = Task.init(
        project_name="ASL_recognition",
        task_name="data validation",
        task_type=TaskTypes.data_processing,
    )
    clearml_params = {"dataset_id": "4f15d3acaec34093b3dc51f42cdf2539"}
    task.connect(clearml_params)
    task.execute_remotely()
    dataset_path = Dataset.get(**clearml_params).get_local_copy()

    config: AppConfig = AppConfig.parse_raw(filename=config_path)
    config.dataset_path = Path(dataset_path)
    main_actions(config=config)


if __name__ == "__main__":
    main()
