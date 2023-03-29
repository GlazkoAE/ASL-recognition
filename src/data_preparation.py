from pathlib import Path

import splitfolders
from PIL import Image
from tqdm import tqdm

from config.config import AppConfig


def prepare_data(config: AppConfig):
    images_path_list = [x for x in config.dataset_path.glob("**/*.jpg")]

    for image_path in tqdm(images_path_list, desc="Images preparation"):
        Image.open(image_path).resize(size=config.imsize).save(image_path)

    train_test_split(config)


def get_class_num(config: AppConfig):
    dataset_path = Path(config.dataset_output_path)
    train_path = dataset_path / "train"
    class_num = 0
    for class_dir in train_path.iterdir():
        if class_dir.is_dir():
            class_num += 1
    return class_num


def train_test_split(config: AppConfig):
    print("Splitting dataset to train/val/test")
    splitfolders.ratio(
        input=config.dataset_path,
        output=config.dataset_output_path,
        seed=config.random_state,
        ratio=config.dataset_split_ratio,
        group_prefix=None,
        move=False,
    )
    print("Done")


def main():
    config = AppConfig.parse_raw()
    prepare_data(config=config)


if __name__ == "__main__":
    main()
