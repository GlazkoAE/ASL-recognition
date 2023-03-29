from torchvision import models


def get_model(model_name: str):
    return getattr(models, model_name)(pretrained=True)
