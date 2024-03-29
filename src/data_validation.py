from PIL import Image
from tqdm import tqdm

from config.config import AppConfig


def validate_data(config: AppConfig):
    images_path_list = [x for x in config.dataset_path.glob("**/*.jpg")]
    for image_path in tqdm(images_path_list, desc="Images validation"):
        Image.open(image_path)


def main():
    config = AppConfig.parse_raw()
    validate_data(config=config)


if __name__ == "__main__":
    main()
