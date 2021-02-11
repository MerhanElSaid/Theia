import torch.hub
import torch.nn as nn


def loadAgeModel():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)

    for name, param in model.named_parameters():
        param.requires_grad = False
    num_classes = 33

    model.fc = nn.Sequential(nn.Linear(model.fc.in_features,512), nn.ReLU(), nn.Dropout(), nn.Linear(512, num_classes))
    return model
