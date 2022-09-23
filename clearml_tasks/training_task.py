import sys
from pathlib import Path

sys.path.append("./..")


def main(config_path="../config/config.yaml"):
    from clearml import Dataset, Task, TaskTypes

    from config.config import AppConfig
    from src.training import main_actions

    config: AppConfig = AppConfig.parse_raw(filename=config_path)

    task: Task = Task.init(
        project_name=config.project_name,
        task_name="training",
        task_type=TaskTypes.training,
    )

    clearml_params = config.dict()
    task.connect(clearml_params)
    task.execute_remotely()
    dataset_path = Dataset.get(clearml_params["dataset_id"]).get_local_copy()
    config.training_dataset_path = Path(dataset_path)
    config.class_num = clearml_params["class_num"]
    main_actions(config=config)


if __name__ == "__main__":
    main()
