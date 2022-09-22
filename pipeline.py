from clearml import PipelineController

from config.config import AppConfig


def pre_execute_callback_example(a_pipeline, a_node, current_param_override):
    # type (PipelineController, PipelineController.Node, dict) -> bool
    print(
        "Cloning Task id={} with parameters: {}".format(
            a_node.base_task_id, current_param_override
        )
    )
    # if we want to skip this node (and subtree of this node) we return False
    # return True to continue DAG execution
    return True


def post_execute_callback_example(a_pipeline, a_node):
    # type (PipelineController, PipelineController.Node) -> None
    print("Completed Task id={}".format(a_node.executed))
    # if we need the actual executed Task: Task.get_task(task_id=a_node.executed)
    return


def run_pipe():
    config: AppConfig = AppConfig.parse_raw()

    pipe = PipelineController(
        name="Training with data preparation pipeline",
        project=config.project_name,
        version="0.0.1",
    )

    pipe.set_default_execution_queue("default")

    pipe.add_step(
        name="validation_data",
        base_task_project=config.project_name,
        base_task_name="data validation",
        parameter_override={"General/dataset_id": config.dataset_id},
        pre_execute_callback=pre_execute_callback_example,
        post_execute_callback=post_execute_callback_example,
    )

    pipe.add_step(
        name="preparation_data",
        parents=[
            "validation_data",
        ],
        base_task_project=config.project_name,
        base_task_name="data preparation",
        parameter_override={"General/dataset_id": config.dataset_id},
        pre_execute_callback=pre_execute_callback_example,
        post_execute_callback=post_execute_callback_example,
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
        pre_execute_callback=pre_execute_callback_example,
        post_execute_callback=post_execute_callback_example,
    )

    # Starting the pipeline (in the background)
    pipe.start_locally()

    print("done")


if __name__ == "__main__":
    run_pipe()