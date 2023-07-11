# © Roel Duijsings
from torch.utils.data import Dataset
from PIL import Image
import clip
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CLIPDataset(Dataset):
    """
    Custom Dataset for loading CLIP model inputs.

    Inherits from the PyTorch Dataset class. Designed to load images and 
    labels for use in a CLIP model, with pre-processing applied to the images.

    Attributes
    ----------
    image_paths : list
        A list of paths to the image files.
    labels : list
        A list of labels corresponding to each image.
    # NOTE: preprocess is also done in CLIPImageClassifier class. Might have to delete one of them?
    preprocess : callable
        The preprocessing function to apply to images before inputting them into the model.
    """
    def __init__(self, image_paths, labels, preprocess):
        """
        Initialize the CLIPDataset.

        Parameters
        ----------
        image_paths : list
            A list of paths to the image files.
        labels : list
            A list of labels corresponding to each image.
        preprocess : callable
            The preprocessing function to apply to images before inputting them into the model.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.preprocess = preprocess

    def __getitem__(self, index):
        """
        Get a single item from the dataset at the specified index.

        Parameters
        ----------
        index : int
            The index to retrieve.

        Returns
        -------
        tuple
            A tuple containing the preprocessed image and its corresponding label tokenized.
        """
        image = self.preprocess(Image.open(self.image_paths[index])) # NOTE: might want to remove preprocess here
        label = clip.tokenize(self.labels[index])[0]
        return image, label

    def __len__(self):
        """
        Get the total number of items in the dataset.

        Returns
        -------
        int
            The total number of items in the dataset.
        """
        return len(self.image_paths)