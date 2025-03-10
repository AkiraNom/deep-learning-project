import torch
from torch import nn
from torchvision.models import mobilenet_v3_large
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights

def load_pretrained_model(base_model=mobilenet_v3_large, weights=MobileNet_V3_Large_Weights, num_classes=102, params_path=None):
    """
    Load a pre-trained model and change the classifier to the number of classes in the dataset.
    """

    # load model
    if params_path:
        model = base_model()
    else:
        model = base_model(weights=weights)

    # change 1000 classes classifier to 102 classes classifier
    model.classifier[-1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)

    # load trained parameters
    model.load_state_dict(torch.load(params_path, map_location=torch.device('cpu')))
    return model


def predict_class(model, image, device='cpu'):
    """
    Predict the class of the flower image

    return the probabilities of the classes
    """

    model.eval()

    with torch.no_grad():
        image = image.to(device)
        preds = model(image)
        probs = torch.nn.functional.softmax(preds, dim=1)

    return probs
