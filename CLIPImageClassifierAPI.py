# © RoelDuijsings
import json
import os
import requests
import torch
from torch.utils.data import DataLoader, Dataset
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import (get_choice, get_env, get_local_path,
                                   get_single_tag_keys, is_skipped)
from PIL import Image
import clip
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

UNLABALED_DIR = r"Data\Unlabeled"
LABELED_DIR = r"Data\Labeled"

HOSTNAME = get_env('HOSTNAME', 'http://localhost:8080')
API_KEY = "003a52aa51e843ba009a78636dc3f6ca62023da4"
# API_KEY = get_env("KEY")

print('=> LABEL STUDIO HOSTNAME = ', HOSTNAME)
print('=> API_KEY = ', API_KEY)
if not API_KEY:
    print('=> WARNING! API_KEY is not set')

def convert_to_local_path(ls_path):
    # extract image name and join with unlabeled dir to get local image_path
    image_name= ls_path.split('-')[1]
    image_path = os.path.join(UNLABALED_DIR, image_name)
    return image_path

def calculate_uncertainty(top_probs):
    # SMALLEST MARGIN UNCERTAINTY (SMU) P(max)-P(max-1)
    first = top_probs[0][0].item()
    second = top_probs[0][1].item()
    margin = first - second
    return 1 - margin


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

class CLIPImageClassifier(object):
    """ 
    Image classifier using the CLIP model for use in a (Label Studio) Active Learning loop.
    
    Attributes
    ----------
    model : torch.nn.Module
        The CLIP model used for classification.
    preprocess : callable
        The preprocessing function to apply to images before inputting them into the model.
    loss_img : torch.nn.CrossEntropyLoss
        The loss function used for images.
    loss_txt : torch.nn.CrossEntropyLoss
        The loss function used for texts.
    optimizer : torch.optim.Optimizer
        The optimizer used for training the model.
    """
    def __init__(self, learning_rate):#, T_max:int=0):
        """
        Initialize the CLIPImageClassifier.

        Parameters
        ----------
        learning_rate : float
            The learning rate to use in the Adam optimizer for training.
        """
        self.model, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)
        self.model = self.model.to(device)
        
        # Define a loss function for the images and texts
        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()
        
        # Define an optimizer and a scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max) #T_max = len(dataloader)*self.num_epochs) #NOTE: scheduler
    
    def save(self, path):
        """
        Save the current state of the model.

        Parameters
        ----------
        path : str
            The path where the model state should be saved.
        """
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        """
        Load a saved state of the model.

        Parameters
        ----------
        path : str
            The path from where the model state should be loaded.
        """
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        
    def predict(self, image_path, labels):
        """
        Predict the class of an image using the model.

        Parameters
        ----------
        image_path : str
            The path of the image to classify.
        labels : list
            The list of labels (texts) for the images.

        Returns
        -------
        tuple
            A tuple containing the logits for the image and the text.
        """ 
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(device) # NOTE: now image per image, but can insert multiple images as a list to the model. The problem you're facing is related to the fact that your CLIP model is expecting a batch of images, but you're giving it a single image.In PyTorch, models expect inputs to have a batch dimension, even if there's only one item in the batch. In other words, if you're providing a single image, it still needs to be presented as a batch of size 1.
        logits_per_image, logits_per_text = self.model(image, labels)
        return logits_per_image.to(device), logits_per_text.to(device)
    
    def train(self, dataloader, num_epochs=20):
        """
        Train the model.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            The DataLoader providing the training data.
        num_epochs : int, optional
            The number of epochs to train for (default is 20).
        """
        since = time.time()

        #https://github.com/openai/CLIP/issues/57
        def convert_models_to_fp32(model): 
            for p in model.parameters(): 
                p.data = p.data.float() 
                p.grad.data = p.grad.data.float() 

        self.model.train()
        for epoch in tqdm(range(num_epochs)):
            # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            # print('-' * 10)
            
            running_loss = 0.0

            # Iterate over data.
            for batch in dataloader:
                self.optimizer.zero_grad()

                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

                # NOTE: now it inputs a list of images, instead of single images
                logits_per_image, logits_per_text = self.model(images, labels)
                ground_truth = torch.arange(len(images), device=device)

                total_loss = (self.loss_img(logits_per_image, ground_truth) + self.loss_txt(logits_per_text,ground_truth)) / 2
                total_loss.backward()

                if device == "cpu":
                    self.optimizer.step()
                else : 
                    convert_models_to_fp32(self.model)
                    self.optimizer.step()
                    clip.model.convert_weights(self.model)

                # loss statistics
                running_loss += total_loss.item() * images.size(0)
                # self.scheduler.step()

            # loss per epoch
            epoch_loss = running_loss / len(dataloader.dataset)
            # print(' Train Loss: {:.4f}'.format(epoch_loss))

        time_elapsed = time.time() - since
        # print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

class CLIPImageClassifierAPI(LabelStudioMLBase):
    # TODO: integrate CLIPImageClassifier
    """ 
    CLIP Image Classifier API connector for Label Studio's Active Learning loop.

    This class is designed to interact with Label Studio's machine learning backend and 
    integrates the CLIPImageClassifier class for image classification tasks. It uses the 
    LabelStudioMLBase class from Label Studio as a base class, and adds additional functionality 
    for interacting with the CLIP image classification model.

    Attributes
    ----------
    model : CLIPImageClassifier
        The instance of the CLIPImageClassifier used for image classification tasks.
    from_name : str
        The source field of the annotation.
    to_name : str
        The target field of the annotation.
    value : str
        The value of the annotation field.
    classes : list
        The list of class labels for classification.

    Methods
    -------
    predict(tasks, **kwargs)
        Predict the labels of given tasks.
    fit(annotations, workdir=None, batch_size=10, num_epochs=10, **kwargs)
        Train the model on given annotations.
    _get_annotated_dataset(project_id)
        Retrieve annotated data from Label Studio API.
    reset_model()
        Reset the model for training.
    """
    def __init__(self, **kwargs):
        """
        Initialize the CLIPImageClassifierAPI with default settings.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments.
        """
        super(CLIPImageClassifierAPI, self).__init__( **kwargs)
        self.model = CLIPImageClassifier()
        
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, 'Choices', 'Image')
        
    def predict(self, tasks, **kwargs):
        """
        Predict the labels of given tasks using the trained model.

        Parameters
        ----------
        tasks : list
            A list of tasks for prediction.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        list
            A list of prediction results and scores for each task.
        """
        print("Start predicting!")
        labels = self.classes
        labels = clip.tokenize(labels).to(device)

        predictions = []
        predicted_labels = []
        with torch.no_grad():
            for task in tasks:
                ls_image_path = task['data'][self.value]
                image_path = convert_to_local_path(ls_image_path)
                
                logits_per_image, logits_per_label = self.model.predict(image_path, labels)

                probs = logits_per_image.softmax(dim=-1) #take the softmax to get the label probabilities
                k = min(5, len(self.classes))
                top_probs, top_labels_indices = probs.topk(k=k, dim=-1) # returns values,indices

                topk_labels = [self.classes[id] for id in top_labels_indices[0]] # the top k predicted labels

                top_score = top_probs[0][0].item()
                predicted_label = topk_labels[0]

                predicted_labels.append(predicted_label)

                result = [{
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'choices',
                    'value': {'choices': [predicted_label]},
                    # 'topk_labels': topk_labels,
                }]
                
                # The Predicition score in LabelStudio is currently the uncertainty score calculated below
                uncertainty_score = calculate_uncertainty(top_probs)
                predictions.append({'result': result, 'score': uncertainty_score}) #'score': float(top_score)})

        with open("predictions.txt", 'w') as f:
            for label in predicted_labels:
                f.write(f"{label}\n")

        print("Finished predicting!")
        return predictions

    def fit(self, annotations, workdir=None, batch_size=10, num_epochs=10, **kwargs):
        """
        Train the model on the given annotations.

        Parameters
        ----------
        annotations : list
            A list of annotations for training.
        workdir : str, optional
            The directory to save the trained model.
        batch_size : int, optional
            The size of the batch for training.
        num_epochs : int, optional
            The number of epochs for training.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        dict
            A dictionary containing the path to the trained model and the list of class labels.
        """
        image_paths, image_labels = [], []
        print('Collecting annotations...')
        
        # check if training is from webhook
        if kwargs.get('data'):
            project_id = kwargs['data']['project']['id']
            tasks = self._get_annotated_dataset(project_id)
        # ML training without web hook
        else:
            tasks = annotations
        
        # extract image paths
        for task in tasks:
            # only add labeled images to dataset
            if not task.get('annotations'):
                continue
            annotation = task['annotations'][0] # get input text from task data
            if annotation.get('skipped') or annotation.get('was_cancelled'):
                continue
            
            ls_path =  task['data']['image']
            image_path = convert_to_local_path(ls_path)
            image_paths.append(image_path)           
            image_labels.append(annotation['result'][0]['value']['choices'][0])
        
        print(f'Creating dataset with {len(image_paths)} images...')
        dataset = CLIPDataset(image_paths, image_labels, self.model.preprocess)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

        print('Train model...')
        self.reset_model()
        self.model.train(dataloader, num_epochs=num_epochs)

        print('Save model...')
        model_path = os.path.join(workdir, 'model.pt')
        self.model.save(model_path)
        print("Finish saving.")
        print("--- Finished training! ---")

        return {'model_path': model_path, 'classes': dataset.classes}

    def _get_annotated_dataset(self, project_id):
        """
        Retrieve annotated data from Label Studio API for a given project.

        Parameters
        ----------
        project_id : int
            The id of the project.

        Returns
        -------
        list
            A list of tasks retrieved from Label Studio API.
        """        
        download_url = f'{HOSTNAME.rstrip("/")}/api/projects/{project_id}/export'
        response = requests.get(download_url, headers={'Authorization': f'Token {API_KEY}'})
        if response.status_code != 200:
            raise Exception(f"Can't load task data using {download_url}, "
                            f"response status_code = {response.status_code}")
        return json.loads(response.content)
   
    def reset_model(self):
        """
        Reset the model for training.

        Returns
        -------
        None
        """
        self.model = CLIPImageClassifier()#len(self.classes), self.freeze_extractor)
