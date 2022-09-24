from clearml import Dataset


def load_dataset(clearml_params):
    if clearml_params["dataset_id"]:
        dataset = Dataset.get(dataset_id=clearml_params["dataset_id"])
        dataset_path = dataset.get_local_copy()
    else:
        dataset = Dataset.get(
            dataset_name=clearml_params["dataset_name"],
            dataset_tags=["latest"],
        )
        dataset_path = dataset.get_local_copy()
    return dataset_path, dataset.id
