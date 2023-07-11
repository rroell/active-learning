# © Roel Duijsings
import time
import pandas as pd
from tqdm import tqdm
from CLIPDataset import CLIPDataset
from CLIPImageClassifier import CLIPImageClassifier
from torch.utils.data import DataLoader
import clip
import torch
import csv


def train(images: list, labels: list):
    """
    Trains the model on the provided images and corresponding labels. 

    After training, the model is saved at the path specified by the global variable MODEL_PATH.

    Parameters
    ----------
    images : list
        A list of paths for the images to be used for training. Each element of the list is expected to be a string representing a valid path to an image file.
    
    labels : list
        A list of labels corresponding to the images provided. Each element of the list is expected to be the correct class or label of the corresponding image in the 'images' list.

    Note
    ----
    This function assumes the existence of certain global variables, including MODEL_PATH, BATCH_SIZE, and NUM_EPOCHS, which represent the path where the model should be saved, the batch size to be used during training, and the number of training epochs, respectively.
    """
    dataset = CLIPDataset(images, labels, model.preprocess)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)

    model.train(dataloader, NUM_EPOCHS)
    model.save(MODEL_PATH)

def predict(df_to_predict: pd.DataFrame, unique_annotations: list):
    """
    Make a prediction on the validation images.
    
    Parameters
    ----------
    df_to_predict : pd.DataFrame
        The DataFrame containing the validation images to make predictions on. Each row represents an imagepath and true_label.
    
    unique_annotations : list
        The unique annotations that the model can predict. This is a list of the possible classes or labels in the classification problem.

    Returns
    -------
    top1 : float
        The top-1 accuracy of the model's predictions. This is the proportion of images for which the model's highest-ranked prediction was the correct class or label.
    
    top5 : float
        The top-5 accuracy of the model's predictions. This is the proportion of images for which the correct class or label was among the model's top 5 ranked predictions.
    """
    annotations_tokenized = clip.tokenize(unique_annotations).to(device)

    ncorrect_top1, ncorrect_top5 = 0, 0
    with torch.no_grad():
        # for all images, make a prediction
        print(f"Images to predict: {len(df_to_predict)}.")

        # TODO: replace iterrows of df_to_predict for a more efficient option: list of paths and list of true_labels
        for idx, row in tqdm(df_to_predict.iterrows(), total=df_to_predict.shape[0]):

            path = row[COLUMN_PATH]
            true_label = row[COLUMN_TRUE_LABEL]

            logits_per_image, _ = model.predict(path, annotations_tokenized)
            probs = logits_per_image.softmax(dim=-1) #take the softmax to get the label probabilities

            top_probs, top_labels_indices = probs.topk(k=5, dim=-1) # returns values,indices
            topk_labels = [unique_annotations[id] for id in top_labels_indices[0]] # the top k predicted labels
            predicted_label = topk_labels[0]

            # accuracy per image
            if true_label==predicted_label: 
                ncorrect_top1+=1
                ncorrect_top5+=1
            elif true_label in topk_labels:
                ncorrect_top5+=1
    
    # iteration accuracy
    top1 = round(ncorrect_top1/len(df_to_predict), 5)
    top5 = round(ncorrect_top5/len(df_to_predict), 5)
    
    # print("Finish predicting.")
    return top1, top5
       
def write_results(run, iteration, n_annotations, top1_val, top5_val, n_annotated_classes, lr):
    """
    Write the results to a csv file specified by global variable RESULTS_FILE and print the results to the terminal.

    Parameters
    ----------
    run : int
        The current run number. Used to track multiple runs.
    iteration : int
        The current iteration number in the current run.
    n_annotations : int
        The number of annotated samples used for training in the current iteration.
    top1_val : float
        The Top-1 accuracy of the model on the validation set for the current iteration.
    top5_val : float
        The Top-5 accuracy of the model on the validation set for the current iteration.
    n_annotated_classes : int
        The number of classes that have at least one annotated sample in the training set.
    lr : float
        The learning rate used for training in the current iteration.
        
    Returns
    -------
    None
    """
    print(f"- - Results run {run} iteration {iteration} - -")
    print(f"Train size: {n_annotations}")
    print(f"Top1_val: {top1_val}     Top5_val:{top5_val} ")

    # Write results to csv
    with open(RESULTS_FILE, 'a') as f:
        writer = csv.writer(f)
        res = [run, iteration, n_annotations, top1_val, top5_val, n_annotated_classes, "{:.0e}".format(lr)]
        writer.writerow(res)


######## START RUN ########
from sklearn.model_selection import train_test_split

device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use CUDA's mixed precision training.

# HYPERPARAMETERS

RAW_DATA = 'fashion_data_top40classes.csv'
COLUMN_TRUE_LABEL = 'product_sub_category'
COLUMN_PATH = 'product_image_path'
MODEL_PATH = 'fashion_model.pt'

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5
N_VAL = 200
N_PER_CLASS = 3
RESULTS_FILE = r'results\fashiontop40_results_noAL.csv'

# Read the top40 classes dataset
df = pd.read_csv(RAW_DATA, index_col='product_id') #TEMP
print(f"df shape: {df.shape}")

# Split the data into a training set and a validation set
df_train, df_val= train_test_split(df, test_size=0.2, random_state=42)
print(f"Train data shape {df_train.shape}")
print(f"Validation data shape {df_val.shape}")

RUNS = 10
ITERATIONS = 10

for run in range(RUNS):
    start_time = time.time()

    for iteration in range(ITERATIONS):
        print(f"\n--Start run {run} iteration {iteration}")
        model = CLIPImageClassifier(LEARNING_RATE)

        # iteration zero: stratified sampling: select 3 IMAGES PER CLASS, SO EVERY CLASS IS REPRESENTED
        df_samples = df_train.groupby(COLUMN_TRUE_LABEL).apply(lambda c: c.sample(N_PER_CLASS)) # sample 3 images from each class
        df_samples.index = df_samples.index.droplevel(0)
        
        # add random samples for the next iterations
        if iteration > 0:
            additional_samples = df_train.drop(df_samples.index).sample(n=10*iteration)  # add 10, 20, 30... samples
            df_samples = pd.concat([df_samples, additional_samples])
        
        # start training on the annotated images
        annotated_classes = df[COLUMN_TRUE_LABEL].unique().tolist()
        print(f"Train on {df_samples.shape[0]} annotated samples from {len(annotated_classes)} classes for {NUM_EPOCHS} epochs.")
        images = df_samples[COLUMN_PATH].tolist()
        labels = df_samples[COLUMN_TRUE_LABEL].tolist()
        train(images, labels)

        # validation 
        print("--Validation--") 
        df_to_predict_val = df_val.sample(n=N_VAL)
        top1_val, top5_val = predict(df_to_predict_val, annotated_classes) 

        write_results(run, iteration, len(df_samples), top1_val, top5_val, len(annotated_classes), LEARNING_RATE)

    time_elapsed = time.time() - start_time
    print('\nTraditional ML completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print("End.")