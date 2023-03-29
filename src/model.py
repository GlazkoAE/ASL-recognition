from pathlib import Path
import pathlib

import sklearn.metrics as metrics
import torch
from torch import nn

from src.utils import get_model


class Model:
    def __init__(self, model_name, optimizer, loss_func, lr, class_num, seed):
        torch.manual_seed(seed)
        self.model = get_model(model_name)
        self._freeze_layers()
        self._setup_fc_layer(class_num)
        self._setup_optimizer(optimizer, lr)
        self.loss_func = loss_func
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def training_step(self, dataloader, writer, epoch, log_every):
        running_loss = 0

        self.model.train()
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            output = self.model(images)
            loss = self.loss_func(output, labels)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            # Log
            if batch_idx % log_every == log_every - 1:
                last_loss = running_loss / log_every
                print(
                    "Epoch {0}, batch {1}/{2}: train loss {3}".format(
                        epoch, batch_idx + 1, len(dataloader), last_loss
                    )
                )
                niter = epoch * len(dataloader) + batch_idx + 1
                writer.add_scalar("Train/Loss", last_loss, niter)
                running_loss = 0

    def validation_step(self, dataloader, writer, epoch):
        running_loss = 0
        running_acc = 0
        running_f1_micro = 0
        running_f1_macro = 0
        running_f1_w = 0

        print("Epoch {0} validation...".format(epoch))
        self.model.eval()
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            output = self.model(images)
            loss = self.loss_func(output, labels)
            running_loss += loss.item()

            # Metrics
            labels = labels.cpu().data.numpy()
            output_max = torch.argmax(output.cpu(), dim=1).data.numpy()
            running_acc += metrics.accuracy_score(labels, output_max)
            running_f1_micro += metrics.f1_score(labels, output_max, average="micro")
            running_f1_macro += metrics.f1_score(labels, output_max, average="macro")
            running_f1_w += metrics.f1_score(labels, output_max, average="weighted")

        avg_loss = running_loss / len(dataloader)
        print("Val loss ", avg_loss)
        writer.add_scalar("Val/Loss", avg_loss, epoch + 1)
        writer.add_scalar(
            "Val_metrics/Accuracy", running_acc / len(dataloader), epoch + 1
        )
        writer.add_scalar(
            "Val_metrics/F1_micro", running_f1_micro / len(dataloader), epoch + 1
        )
        writer.add_scalar(
            "Val_metrics/F1_macro", running_f1_macro / len(dataloader), epoch + 1
        )
        writer.add_scalar(
            "Val_metrics/F1_weighted", running_f1_w / len(dataloader), epoch + 1
        )

        return avg_loss

    def serialize(self, model_path, image_size, name):
        # Initialize model with the pretrained weights
        self.model.load_state_dict(torch.load(model_path))

        # set the model to inference mode
        self.model.to(torch.device("cpu"))
        self.model.eval()

        # Input to the model
        x = torch.randn(1, 3, image_size[0], image_size[1], requires_grad=True)
        onnx_path = Path(pathlib.__file__).parent / (name + ".onnx")

        # Export the model
        torch.onnx.export(
            self.model,  # model being run
            x,  # model input
            onnx_path,
            export_params=True,
            opset_version=10,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
        )
        return onnx_path

    def _freeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def _setup_fc_layer(self, class_num):
        in_features = self.model.fc.in_features
        fc = nn.Linear(in_features=in_features, out_features=class_num)
        self.model.fc = fc

    def _setup_optimizer(self, optimizer, lr):
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        self.optimizer = optimizer(params_to_update, lr=lr)
