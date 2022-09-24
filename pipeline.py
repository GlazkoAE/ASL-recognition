import argparse

from clearml import PipelineController

from config.config import AppConfig


def run_pipe(is_local):
    config: AppConfig = AppConfig.parse_raw()

    pipe = PipelineController(
        name="Full pipeline",
        project="ASL_recognition",
        version="0.1.0",
    )

    pipe.add_parameter("dataset_id", "")

    pipe.set_default_execution_queue("default")

    pipe.add_step(
        name="validation_data",
        base_task_project="ASL_recognition",
        base_task_name="data validation",
        parameter_override={"General/dataset_id": "${pipeline.dataset_id}"},
    )

    pipe.add_step(
        name="preparation_data",
        parents=[
            "validation_data",
        ],
        base_task_project="ASL_recognition",
        base_task_name="data preparation",
        parameter_override={
            "General/dataset_id": "${validation_data.parameters.General/dataset_id}"
        },
    )

    pipe.add_step(
        name="training_step",
        parents=[
            "preparation_data",
        ],
        base_task_project="ASL_recognition",
        base_task_name="training",
        parameter_override={
            "General/dataset_id": "${preparation_data.parameters.General/output_dataset_id}",
            "General/class_num": "${preparation_data.parameters.General/class_num}",
            "General/dataset_name": "${preparation_data.parameters.General/output_dataset_name}",
        },
    )

    if is_local:
        pipe.start_locally(run_pipeline_steps_locally=True)
    else:
        pipe.start(queue="default")
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local", action="store_const", dest="is_local", const=True, default=False
    )
    args = parser.parse_args()

    run_pipe(args.is_local)
