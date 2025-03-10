
from PIL import Image
import json
import plotly.express as px
import torch
from torchvision import transforms

class ImageProcessing:
    @staticmethod
    def load_image(image_path):
        """
        Load the image from the path
        """

        # image = plt.imread(image_path)
        image = Image.open(image_path).convert('RGB')

        return image

    def preprocess_iamge(image, img_size=128):
        """
        iamge is preprocessed to be fed into the model
        """

        # Define the preprocessing transformations
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2))
        ])

        # Apply the transformations
        input_tensor = transform(image)

        # Add batch dimension
        input_batch = input_tensor.unsqueeze(0)  # Shape: (1, 3, 224, 224)

        return input_batch

    @staticmethod
    def display_image(image):
        """
        Display the image
        """

        fig = px.imshow(image)
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        return fig

# path for local deployment
# file_path = './img/cat_to_name.json'

# path for deployment on the streamlit cloud
file_path = './deep-learning-project/CNN-Transfer-Learning/Flower_image_classification/img/cat_to_name.json'
def load_category_names(file_path=file_path):
    """
    Load the category names from the json file
    """

    with open(file_path, 'r') as f:
        cat_to_name = json.load(f)

    return  cat_to_name

class ResultAnalysis:

    @staticmethod
    def find_top_classes(probs, classes, topk=5):
        """
        Find the top k classes with their probabilities
        """

        probs, indices = torch.topk(probs, topk)

        probs_list = probs[0].cpu().tolist()
        indices_list = indices[0].cpu().tolist()

        class_name = [classes[str(index+1)] for index in indices_list]

        # sort the classes and probabilities according to the probabilities
        probs_list, class_name = zip(*sorted(zip(probs_list, class_name)))

        plot_title = f'Top {topk} Class Probabilities'

        return probs_list, class_name, plot_title

    @staticmethod
    def sanity_check(probs:list, classes:list, plot_title=''):
        """
        Display the top k classes with their probabilities
        """
        fig = px.bar(x=probs, y=classes, orientation='h', title=plot_title, labels={'x':'Probability','y':'Class'})

        return fig
