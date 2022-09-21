import os

from PIL import Image
from tqdm import tqdm

from config.config import AppConfig


def main_actions(config: AppConfig):
    images_path_list = [x for x in config.dataset_path.glob("**/*.jpg")]
    dataset_path = config.dataset_output_path
    dataset_path.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(images_path_list, desc="Images preparation"):
        class_name = image_path.parts[-2]  # Folder Name
        stage_name = image_path.parts[-3]  # Train/Test/Val
        class_folder = dataset_path / stage_name / class_name
        class_folder.mkdir(parents=True, exist_ok=True)
        Image.open(image_path).resize(size=config.imsize).save(
            class_folder / image_path.name
        )


def get_class_num(config: AppConfig):
    dataset_path = config.dataset_output_path
    train_path = os.path.join(dataset_path, "train")
    class_num = 0
    for class_dir in os.listdir(train_path):
        if os.path.isdir(os.path.join(train_path, class_dir)):
            class_num += 1
    return class_num


def main():
    config = AppConfig.parse_raw()
    main_actions(config=config)


if __name__ == "__main__":
    main()
