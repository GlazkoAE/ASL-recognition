from clearml import Task, TaskTypes

from config.config import AppConfig
from src.triton_server_preparation import build_triton_server


def main(config_path="./config/config.yaml"):
    config: AppConfig = AppConfig.parse_raw(filename=config_path)

    task: Task = Task.init(
        project_name=config.project_name,
        task_name="build_triton_server",
        task_type=TaskTypes.inference,
    )

    clearml_params = {
        "project_name": config.project_name,
        "training_task_id": config.training_task_id,
        "http_endpoint": config.http_endpoint,
        "grpc_endpoint": config.grpc_endpoint,
        "prometheus_endpoint": config.prometheus_endpoint,
    }
    task.connect(clearml_params)

    training_task = Task.get_task(task_id=clearml_params["training_task_id"])

    clearml_params["class_num"] = training_task.get_parameter("General/class_num")
    clearml_params["image_size"] = training_task.get_parameter("General/image_size")
    clearml_params["model"] = training_task.get_parameter("General/model")
    task.set_parameters_as_dict(clearml_params)

    config.class_num = clearml_params["class_num"]
    config.imsize = tuple(map(int, clearml_params["image_size"][1:-1].split(", ")))
    config.model = clearml_params["model"]

    model = training_task.artifacts["onnx_model"].get_local_copy()

    build_triton_server(config=config, model=model)

    task.upload_artifact(name="server_model", artifact_object="./../triton/server/")


if __name__ == "__main__":
    main()
