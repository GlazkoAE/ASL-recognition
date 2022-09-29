from torchvision import models


def get_model(model_name):
    if model_name == "alexnet":
        return models.alexnet(pretrained=True)
    elif model_name == "vgg11":
        return models.vgg11(pretrained=True)
    elif model_name == "vgg11_bn":
        return models.vgg11_bn(pretrained=True)
    elif model_name == "vgg13":
        return models.vgg13(pretrained=True)
    elif model_name == "vgg13_bn":
        return models.vgg13_bn(pretrained=True)
    elif model_name == "vgg16":
        return models.vgg16(pretrained=True)
    elif model_name == "vgg19":
        return models.vgg19(pretrained=True)
    elif model_name == "vgg19_bn":
        return models.vgg19_bn(pretrained=True)
    elif model_name == "resnet18":
        return models.resnet18(pretrained=True)
    elif model_name == "resnet34":
        return models.resnet34(pretrained=True)
    elif model_name == "resnet50":
        return models.resnet50(pretrained=True)
    elif model_name == "resnet101":
        return models.resnet101(pretrained=True)
    elif model_name == "resnet152":
        return models.resnet152(pretrained=True)
    elif model_name == "squeezenet1_0":
        return models.squeezenet1_0(pretrained=True)
    elif model_name == "squeezenet1_1":
        return models.squeezenet1_1(pretrained=True)
    elif model_name == "densenet121":
        return models.densenet121(pretrained=True)
    elif model_name == "densenet169":
        return models.densenet169(pretrained=True)
    elif model_name == "densenet161":
        return models.densenet161(pretrained=True)
    elif model_name == "densenet201":
        return models.densenet201(pretrained=True)
    elif model_name == "inception_v3":
        return models.inception_v3(pretrained=True)
    elif model_name == "googlenet":
        return models.googlenet(pretrained=True)
    elif model_name == "shufflenet_v2_x0_5":
        return models.shufflenet_v2_x0_5(pretrained=True)
    elif model_name == "shufflenet_v2_x1_0":
        return models.shufflenet_v2_x1_0(pretrained=True)
    elif model_name == "shufflenet_v2_x1_5":
        return models.shufflenet_v2_x1_5(pretrained=True)
    elif model_name == "shufflenet_v2_x2_0":
        return models.shufflenet_v2_x2_0(pretrained=True)
    elif model_name == "mobilenet_v2":
        return models.mobilenet_v2(pretrained=True)
    elif model_name == "resnext50_32x4d":
        return models.resnext50_32x4d(pretrained=True)
    elif model_name == "resnext101_32x8d":
        return models.resnext101_32x8d(pretrained=True)
    elif model_name == "wide_resnet50_2":
        return models.wide_resnet50_2(pretrained=True)
    elif model_name == "wide_resnet101_2":
        return models.wide_resnet101_2(pretrained=True)
    elif model_name == "mnasnet0_5":
        return models.mnasnet0_5(pretrained=True)
    elif model_name == "mnasnet0_75":
        return models.mnasnet0_75(pretrained=True)
    elif model_name == "mnasnet1_0":
        return models.mnasnet1_0(pretrained=True)
    elif model_name == "mnasnet1_3":
        return models.mnasnet1_3(pretrained=True)
