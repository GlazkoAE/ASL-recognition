import cv2
import torch
from torchvision import datasets


class Dataset:
    def __init__(self, dataset_path, batch_size, num_workers):
        self.image_datasets = {
            x: datasets.ImageFolder(
                dataset_path / x,
                loader=lambda x: torch.Tensor(cv2.imread(x)).permute(2, 0, 1),
            )
            for x in ["train", "val", "test"]
        }

        self.train_dataloader = torch.utils.data.DataLoader(
            self.image_datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        self.val_dataloader = torch.utils.data.DataLoader(
            self.image_datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self.test_dataloader = torch.utils.data.DataLoader(
            self.image_datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self.labels_map = self.image_datasets["train"].find_classes(
            dataset_path / "train"
        )
        self.labels_map = list(self.labels_map[1].keys())
