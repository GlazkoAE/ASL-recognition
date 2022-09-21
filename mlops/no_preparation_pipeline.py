from clearml import PipelineController

from config.config import AppConfig

config: AppConfig = AppConfig.parse_raw()

pipe = PipelineController(
    name="Training pipeline", project=config.project_name, version="0.0.1"
)
pipe.add_step(
    name="validation_data",
    base_task_project=config.project_name,
    base_task_name="data validation",
    parameter_override={"General/dataset_id": config.dataset_id},
)
pipe.add_step(
    name="training_step",
    parents=[
        "validation_data",
    ],
    base_task_project=config.project_name,
    base_task_name="training",
    parameter_override={
        "General/dataset_id": "${validation_data.parameters.dataset_id}"
    },
)

# for debugging purposes use local jobs
pipe.start_locally()

# Starting the pipeline (in the background)
# pipe.start()

print("done")
