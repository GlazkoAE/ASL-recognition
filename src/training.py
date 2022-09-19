import cv2
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torchvision import datasets

from config import AppConfig
from src.utils import get_model


class LightningModel(LightningModule):
    def __init__(self, model, optimizer, loss_func, lr):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.lr = lr

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        logits = self.model(x)
        loss = self.loss_func(logits, y)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer


def main_actions(config: AppConfig):
    seed_everything(config.random_state, workers=True)
    trainer = Trainer(max_epochs=config.epochs, accelerator="gpu")

    image_datasets = {
        x: datasets.ImageFolder(
            config.training_dataset_path / x,
            loader=lambda x: torch.Tensor(cv2.imread(x)),
        )
        for x in ["train", "val"]
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )
        for x in ["train", "val"]
    }

    model_ft = get_model(config.model)
    num_classes = len(dataloaders["train"].dataset.dataset.classes)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
    pl_model = LightningModel(
        model_ft, torch.optim.Adam, torch.nn.CrossEntropyLoss(), config.lr
    )

    trainer.fit(pl_model, dataloaders["train"], dataloaders["val"])


def main():
    pass


if __name__ == "__main__":
    main()
