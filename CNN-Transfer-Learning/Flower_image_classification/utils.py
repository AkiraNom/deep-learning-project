
import Image
import matplotlib.pyplot as plt
from torchvision import transforms

def load_iamge(image_path):
    """
    Load the image from the path
    """

    # image = plt.imread(image_path)
    image = Image.open(image_path).convert('RGB')

    return image

def preprocess_iamge(image, img_size=224):
    """
     iamge is preprocessed to be fed into the model
    """


    # Define the preprocessing transformations
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2))
    ])

    # Apply the transformations to the image
    input_tensor = transform(image)

    # Add batch dimension
    input_batch = input_tensor.unsqueeze(0)  # Shape: (1, 3, 224, 224)

    return input_batch
