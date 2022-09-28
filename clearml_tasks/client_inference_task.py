import sys

sys.path.append("./..")


def main(config_path="../config/config.yaml"):
    from clearml import Task, TaskTypes

    from config.config import AppConfig
    from src.triton_client_preparation import build_triton_client

    config: AppConfig = AppConfig.parse_raw(filename=config_path)

    task: Task = Task.init(
        project_name=config.project_name,
        task_name="run_triton_client",
        task_type=TaskTypes.inference,
    )

    clearml_params = {
        "project_name": config.project_name,
        "training_task_id": config.training_task_id,
        "server_host": config.server_host,
        "grpc_endpoint": config.grpc_endpoint,
    }
    task.connect(clearml_params)

    training_task = Task.get_task(task_id=clearml_params["training_task_id"])

    clearml_params["class_num"] = training_task.get_parameter("General/class_num")
    task.set_parameters_as_dict(clearml_params)

    config.class_num = clearml_params["class_num"]

    labels = training_task.artifacts["labels_map"].get_local_copy()

    build_triton_client(config=config, labels=labels)

    task.upload_artifact(name="client", artifact_object="./../triton/client/")


if __name__ == "__main__":
    main()
