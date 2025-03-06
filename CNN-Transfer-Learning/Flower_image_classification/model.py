import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights

import matplotlib.pyplot as plt



def load_pretrained_model(base_model=mobilenet_v3_small, weights=MobileNet_V3_Small_Weights, num_classes=102, params_path=None):
    """
    Load a pre-trained model and change the classifier to the number of classes in the dataset.
    """

    # load model
    if params_path:
        model = base_model()
    else:
        model = base_model(weights=weights)

    # change 1000 classes classifier to 102 classes classifier
    model.classifier[-1] = nn.Linear(in_features=1024, out_features=num_classes, bias=True)

    # load trained parameters
    model.load_state_dict(torch.load(params_path, map_location=torch.device('cpu')))
    return model


def predict_class(image, model, device='cpu'):
    """
    """

    pass

