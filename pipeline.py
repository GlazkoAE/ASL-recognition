import argparse

from clearml import PipelineController


def run_pipe(is_local):

    pipe = PipelineController(
        name="ASL recognition full pipeline",
        project="ASL_recognition",
        version="1.0.0",
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
        name="training",
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

    pipe.add_step(
        name="server_inference",
        parents=[
            "training",
        ],
        base_task_project="ASL_recognition",
        base_task_name="build_triton_server",
        parameter_override={
            "General/training_task_id": "${training.id}",
        },
    )

    pipe.add_step(
        name="client_inference",
        parents=[
            "training",
        ],
        base_task_project="ASL_recognition",
        base_task_name="build_triton_client",
        parameter_override={
            "General/training_task_id": "${training.id}",
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
