# © Roel Duijsings
import torch
import torch.nn as nn
import torch.optim as optim
import time
import clip
from PIL import Image
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CLIPImageClassifier(object):
    """ 
    CLIP image classifier voor Label Studio Active Learning loop.
    """
    def __init__(self, learning_rate):#, T_max:int=0):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)
        self.model = self.model.to(device)
        
        # Define a loss function for the images and texts
        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()
        
        # Define an optimizer and a scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max) #T_max = len(dataloader)*self.num_epochs) #NOTE: scheduler
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        
    def predict(self, image_path, labels):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(device) # NOTE: now image per image, but can insert multiple images as a list to the model. The problem you're facing is related to the fact that your CLIP model is expecting a batch of images, but you're giving it a single image.In PyTorch, models expect inputs to have a batch dimension, even if there's only one item in the batch. In other words, if you're providing a single image, it still needs to be presented as a batch of size 1.
        logits_per_image, logits_per_text = self.model(image, labels)
        return logits_per_image.to(device), logits_per_text.to(device)
    
    def train(self, dataloader, num_epochs=20):
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

                # NOTE: should this only be loss_img?
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