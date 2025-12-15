from torchvision.models import resnet18
import torch

def get_resnet18_model(num_classes: int):
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model
