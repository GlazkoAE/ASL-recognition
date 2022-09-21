from pathlib import Path

from clearml import Dataset, Task, TaskTypes
from config.config import AppConfig
from src.data_validation import main_actions


def main(config_path="../config/config.yaml"):
    config: AppConfig = AppConfig.parse_raw(filename=config_path)
    task: Task = Task.init(
        project_name=config.project_name,
        task_name="data validation",
        task_type=TaskTypes.data_processing,
    )
    clearml_params = {"dataset_id": config.dataset_id}
    task.connect(clearml_params)
    dataset_path = Dataset.get(**clearml_params).get_local_copy()

    config.dataset_path = Path(dataset_path)
    main_actions(config=config)


if __name__ == "__main__":
    main()
