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


def calculate_uncertainty(top_probs: torch.float16, uncertainty_measure: str) -> float:
    """
    Calculate the uncertainty score for an image using the uncertainty measure. The image is represented as a ranked probability distribution.
    
    Parameters
    ----------
    top_probs : torch.float16
        The top 5 probabilities of the image.

    uncertainty_measure: str
        The uncertainty measure for calculating the uncertainty score.

    Returns
    -------
    float
        The uncertainty score for this image.
    """
    valid_measures = ['margin', 'entropy', 'least', 'random']
    if uncertainty_measure not in valid_measures:
        return f'Invalid uncertainty measure. Please choose one of: {valid_measures}'
    
    # SMALLEST MARGIN UNCERTAINTY (SMU) P(max)-P(max-1)
    if(uncertainty_measure=='margin'):
        first = top_probs[0][0].item()
        second = top_probs[0][1].item()
        margin = first - second
        return 1 - margin
    
    # ENTROPY-BASED UNCERTAINTY
    elif(uncertainty_measure=='entropy'):
        log_probs = torch.log(top_probs)
        entropy = -torch.sum(top_probs * log_probs, dim=-1)
        return entropy.item()
    
    # LEAST CONFIDENCE 
    elif(uncertainty_measure=='least'):
        return 1 - top_probs[0][0].item()
    
    # RANDOM SAMPLING
    elif(uncertainty_measure=='random'):
        return 1

def annotate(n_largest: int, df_train: pd.DataFrame):
    """ 
    Annotate the top n_largest samples with the highest uncertainty score. 
    
    Parameters
    ----------
    n_largest: int
        The number of samples to annotate.

    df_train: pd.DataFrame
        The original training DataFrame.

    Returns
    -------
    df_train : pd.DataFrame
        The updated training DataFrame with the new annotations and uncertainty score column reset to None.
    """

    # Pick top n_largest scored samples and annotate them
    scored_images = df_train.dropna(subset=['score'])
    top_scores = scored_images.nlargest(n_largest, 'score')
    print(f"Annotating {n_largest} images..")

    # here the actual annotating is done: copy the true_label to the annotation coolumn
    df_train.loc[top_scores.index, 'annotation'] = df_train.loc[top_scores.index, COLUMN_TRUE_LABEL]
    df_train['score'] = None       # reset score column for next iteration
    
    # OPTION: save the dataframe containing the updated annotations
    # df_train.to_csv(FILE_NAME, index=False) 
    return df_train

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
    Make a prediction on the images in df_to_predict and calculate their uncertainty score.
    
    Parameters
    ----------
    df_to_predict : pd.DataFrame
        The DataFrame containing the non-annotated images to make predictions on. Each row represents an imagepath and true_label.
    
    unique_annotations : list
        The unique annotations that the model can predict. This is a list of the possible classes or labels in the classification problem.

    Returns
    -------
    df_predictions : pd.DataFrame
        A DataFrame containing the images with columns product_id, uncertainty_score, predicted_label, topk_labels.

    top1 : float
        The top-1 accuracy of the model's predictions. This is the proportion of images for which the model's highest-ranked prediction was the correct class or label.
    
    top5 : float
        The top-5 accuracy of the model's predictions. This is the proportion of images for which the correct class or label was among the model's top 5 ranked predictions.
    """
    annotations_tokenized = clip.tokenize(unique_annotations).to(device)
            
    predictions = []

    ncorrect_top1, ncorrect_top5 = 0, 0
    with torch.no_grad():
        # for all rows that are not annotated yet, make a prediction and calculate the uncertainty score
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

            uncertainty_score = round(calculate_uncertainty(top_probs, UNCERTAINTY_MEASURE), 3)
            
            # predictions_indexes.append(idx)
            predictions.append((idx, uncertainty_score, predicted_label, topk_labels))

            # accuracy per image
            if true_label==predicted_label: 
                ncorrect_top1+=1
                ncorrect_top5+=1
            elif true_label in topk_labels:
                ncorrect_top5+=1
    
    # iteration accuracy
    top1 = round(ncorrect_top1/len(df_to_predict), 5)
    top5 = round(ncorrect_top5/len(df_to_predict), 5)
    
    # Convert the list of predictions to a DataFrame
    df_predictions = pd.DataFrame(predictions, columns=['product_id', 'score', 'prediction', 'top5'])
    df_predictions.set_index('product_id', inplace=True)

    # print("Finish predicting.")
    return df_predictions, top1, top5

def update_df(df_original, df_predictions):
    """
    Update the original training DataFrame by adding the values in df_predictions: {uncertainty score, predicted_label, topk_labels}

    Parameters
    ----------
    df_original : pd.DataFrame
        The original training DataFrame.

    df_predictions : pd.DataFrame
        A DataFrame containing the predictions on the images with columns product_id, uncertainty_score, predicted_label, topk_labels.

    Returns
    -------
    df_original : pd.DataFrame
        The updated training DataFrame.
    """
    for col in ['score', 'prediction', 'top5']:
        df_original.loc[df_predictions.index, col] = df_predictions[col]

    # BUG: Convert 'score' back to float64
    df_original['score'] = df_original['score'].astype('float64')
    return df_original
       
def write_results(run, iteration, n_annotations, top1_train, top5_train, top1_val, top5_val, n_annotated_classes, lr, n_score_per_iteration):
    """
    Write the results to a csv file specified by global variable RESULTS_FILE and print the results to the terminal.
    
    Parameters
    ----------
    run : int
        The current run number.
    iteration : int
        The current iteration number.
    n_annotations : int
        The number of annotations in the training set.
    top1_train : float
        The top-1 accuracy for the training set.
    top5_train : float
        The top-5 accuracy for the training set.
    top1_val : float
        The top-1 accuracy for the validation set.
    top5_val : float
        The top-5 accuracy for the validation set.
    n_annotated_classes : int
        The number of classes with at least one annotation.
    lr : float
        The learning rate used in the current iteration.
    n_score_per_iteration : int
        The number of scores calculated per iteration.
    """

    print(f"- - Results iteration {iteration} - -")
    print(f"Train size: {n_annotations}")
    # print(f"Top1_train: {top1_train}     Top5_train:{top5_train} ")
    print(f"Top1_val: {top1_val}     Top5_val:{top5_val} ")
    # print(f"n_annotated_classes: {n_annotated_classes}")

    # Write results to csv
    with open(RESULTS_FILE, 'a') as f:
        writer = csv.writer(f)
        res = [run, iteration, n_annotations, top1_train, top5_train, top1_val, top5_val, n_annotated_classes, "{:.0e}".format(lr), n_score_per_iteration]
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
N_SCORE_PER_ITERATION = 300
N_ANNOTATE_PER_ITERATION = 10
N_VAL = 200
N_PER_CLASS = 3
UNCERTAINTY_MEASURE = 'random' # valid_measures = ['margin', 'entropy', 'least', 'random']
RESULTS_FILE = fr'results\fashiontop40_results_{UNCERTAINTY_MEASURE}.csv'

# Read the top40 classes dataset
df = pd.read_csv(RAW_DATA, index_col='product_id') #TEMP
print(f"df shape: {df.shape}")
annotated_classes = df[COLUMN_TRUE_LABEL].unique().tolist()

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

        if iteration == 0:
            # for iteration 0, we pick random samples to annotate
            model = CLIPImageClassifier(LEARNING_RATE)

            # OPTION 1 Stratified sampling: 3 images per class, so every class is represented
            df_samples = df_train.groupby(COLUMN_TRUE_LABEL).apply(lambda c: c.sample(N_PER_CLASS)) # sample 3 images from each class
            df_samples.index = df_samples.index.droplevel(0)

            # OPTION 2: randomly select samples, not all classes will be represented
            # df_samples = df_train.sample(random_state=42, n=N_ITERATION_ZERO)

            df_train['annotation'] = df_samples[COLUMN_TRUE_LABEL]  # annotate iteration 0
            
        else:
            # for iterations >0, we annotate based on uncertainty score
            model.load(MODEL_PATH)
            df_train = annotate(N_ANNOTATE_PER_ITERATION, df_train)
        
        # start training on the annotated images
        df_annotated = df_train[df_train['annotation'].notna()]
        print(f"Train on {df_annotated.shape[0]} annotated samples from {len(annotated_classes)} classes for {NUM_EPOCHS} epochs.")
        images = df_annotated[COLUMN_PATH].tolist()
        labels = df_annotated['annotation'].tolist()
        train(images, labels)

        # score the not annotated images
        print("--Score trainingdata--")
        df_unannotated = df_train[df_train['annotation'].isna()]
        df_to_predict_train = df_unannotated.sample(n=N_SCORE_PER_ITERATION) # pick random samples that are not annotated yet, to predict and score them
        df_predictions, top1_train, top5_train = predict(df_to_predict_train, annotated_classes)
        df_train = update_df(df_train, df_predictions)    
        
        # validation 
        print("--Validation--") 
        # calculate validation accuracy, only on images from known classes 
        df_to_predict_val = df_val.sample(n=N_VAL)
        _, top1_val, top5_val = predict(df_to_predict_val, annotated_classes) 

        # print and write results to RESULTS_FILE
        write_results(run, iteration, len(df_annotated), top1_train, top5_train, top1_val, top5_val, len(annotated_classes), LEARNING_RATE, N_SCORE_PER_ITERATION)

    time_elapsed = time.time() - start_time
    print('\nActive Learning completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print("End.")