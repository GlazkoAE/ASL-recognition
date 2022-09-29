import os
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config.config import AppConfig
from src.dataset import Dataset
from src.model import Model


def train_model(config: AppConfig):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter("runs/ASL_trainer_{}".format(timestamp))

    dataset: Dataset = Dataset(
        dataset_path=config.training_dataset_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    model: Model = Model(
        model_name=config.model,
        optimizer=torch.optim.Adam,
        loss_func=torch.nn.CrossEntropyLoss(),
        lr=config.lr,
        class_num=config.class_num,
        seed=config.random_state,
    )

    best_val_loss = 1e6
    model_path = ""

    for epoch in tqdm(range(config.epochs), desc="Model training"):

        model.training_step(
            dataloader=dataset.train_dataloader,
            writer=writer,
            epoch=epoch,
            log_every=config.log_every,
        )

        val_loss = model.validation_step(
            dataloader=dataset.val_dataloader,
            writer=writer,
            epoch=epoch,
        )

        writer.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if model_path:
                os.remove(model_path)
            model_path = "model_{}_{}".format(timestamp, epoch)
            torch.save(model.model.state_dict(), model_path)

    best_model_path = model.serialize(
        model_path=model_path, image_size=config.imsize, name=config.project_name
    )
    os.remove(model_path)

    labels_file = "labels.txt"
    with open(labels_file, "w") as file:
        for item in dataset.labels_map:
            # write each item on a new line
            print(item)
            file.write("%s\n" % item)
    labels_file = os.path.abspath(labels_file)

    labels_file = os.path.abspath(labels_file)

    return best_model_path, labels_file


def main():
    config = AppConfig.parse_raw("./../config/config.yaml")
    train_model(config=config)


if __name__ == "__main__":
    main()
