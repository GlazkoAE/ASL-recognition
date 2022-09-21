from clearml import PipelineController
from config.config import AppConfig


def run_pipe():
    config: AppConfig = AppConfig.parse_raw()

    pipe = PipelineController(
        name="Training with data preparation pipeline", project=config.project_name, version="0.0.1"
    )
    pipe.set_default_execution_queue("default")
    pipe.add_step(
        name="validation_data",
        base_task_project=config.project_name,
        base_task_name="data validation",
        parameter_override={
            "General/dataset_id": config.dataset_id
        },
    )
    pipe.add_step(
        name="preparation_data",
        parents=[
            "validation_data",
        ],
        base_task_project=config.project_name,
        base_task_name="data preparation",
        parameter_override={"General/dataset_id": config.dataset_id},
    )
    pipe.add_step(
        name="training_step",
        parents=[
            "preparation_data",
        ],
        base_task_project=config.project_name,
        base_task_name="training",
        parameter_override={
            "General/dataset_id": "${preparation_data.parameters.output_dataset_id}",
            "General/class_num": "${preparation_data.parameters.class_num}",
        },
    )

    # Starting the pipeline (in the background)
    pipe.start()

    print("done")
