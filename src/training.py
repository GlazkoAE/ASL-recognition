import cv2
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchvision import datasets

from config.config import AppConfig
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
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer


def main_actions(config: AppConfig):
    seed_everything(config.random_state, workers=True)
    trainer = Trainer(
        max_epochs=config.epochs,
        accelerator="gpu",
        callbacks=[EarlyStopping(monitor="val_loss")],
    )

    image_datasets = {
        x: datasets.ImageFolder(
            config.training_dataset_path / x,
            loader=lambda x: torch.Tensor(cv2.imread(x)).permute(2, 0, 1),
        )
        for x in ["train", "val"]
    }

    train_dataloader = torch.utils.data.DataLoader(
        image_datasets["train"],
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    val_dataloader = torch.utils.data.DataLoader(
        image_datasets["val"],
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model_ft = get_model(config.model)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_ftrs, config.class_num)
    pl_model = LightningModel(
        model_ft, torch.optim.Adam, torch.nn.CrossEntropyLoss(), config.lr
    )

    trainer.fit(pl_model, train_dataloader, val_dataloader)


def main():
    pass


if __name__ == "__main__":
    main()
