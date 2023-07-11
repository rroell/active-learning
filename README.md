# Uncertainty-based Active Learning for OpenAI's CLIP Classifier

<p float="left">
  <img 
    src="https://gitlab.com/machine-learning-company/internships/roel-active-learning/-/raw/main/mlc_logo.png" 
    height="100" alt="Squadra MLC" title="Squadra MLC" 
    />
  <img 
    src="https://gitlab.com/machine-learning-company/internships/roel-active-learning/-/raw/main/radboud_logo.png" 
    height="100" alt="Radboud University" title="Radboud University"
    /> 
</p>

This internship project aims to classify images using the CLIP model developed by OpenAI, while implementing an Active Learning strategy based on different uncertainty measures to iteratively select new samples to annotate and add to the training set.

## Structure

The code in this repository is structured as follows:

- `main.py`: This file contains the main logic of the program which includes functions for calculating uncertainty, annotating images, training the model, making predictions, updating dataframes, and writing results. This is also where the active learning strategy is implemented and the training-validation loop resides.

- `main_noAL.py`: Contains the traditional machine learning pipeline, without Active Learning.

- `CLIPImageClassifier.py`: This file contains the `CLIPImageClassifier` class which is used for training and saving the CLIP model, as well as making predictions on unseen images.

- `CLIPImageCLassifierAPI.py`: This file contains the API connector for Label Studio's Active Learning loop. This class is designed to interact with [Label Studio's machine learning backend](https://labelstud.io/guide/ml.html) and
  integrates the CLIPImageClassifier class for image classification tasks.

## Requirements

The required package versions are listed in [requirements.txt](https://gitlab.com/machine-learning-company/internships/roel-active-learning/-/raw/main/requirements.txt).

- Python 3.x
- PyTorch
- torchvision
- tqdm
- pandas
- sklearn
- OpenAI's CLIP

## How to Run

1. Clone this repository
2. Create a virtual environment:
    ```bash
    python -m venv venv
    ```
3. Activate the virtual environment:
    ```bash
    .\venv\Scripts\activate
    ```
    
2. Install requirements using pip:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare your image dataset using `createDataset.py`. The dataset should be in CSV format with columns for image file paths and labels.
4. Adjust the hyperparameters in `main.py`.
5. Run the `main.py` script:
   ```bash
   python main.py
   ```
6. The results will be saved to a CSV file which includes performance metrics such as accuracy for each iteration of the Active Learning strategy.

## Hyperparameters
You can tune the following hyperparameters in the `main.py` file:

- `UNCERTAINTY_MEASURE`: The uncertainty measure used for selecting samples to annotate. Options are 'margin', 'entropy', 'least', and 'random'.
- `N_PER_CLASS`: Number of samples per class for the first iteration.
- `N_SCORE_PER_ITERATION`: Number of samples to score per iteration.
- `N_ANNOTATE_PER_ITERATION`: Number of samples to annotate and rank per iteration.
- `N_VAL`: Number of validation samples.
- `BATCH_SIZE`: Number of samples that going to be propagated through the network.
- `NUM_EPOCHS`: Number of times the entire training set is shown to the network during training.
- `LEARNING_RATE`: The size of the steps the optimizer takes to reach the minimum of the loss function.
- `RESULTS_FILE`: The path where the results will be saved.
- `MODEL_PATH`: The path where the trained model will be saved.

The training-validation loop is executed for a number of runs and iterations, which can be adjusted in the `RUNS` and `ITERATIONS` variables, respectively.

## Label Studio Machine Learning
Integrate the Active Learning pipeline with your data labeling workflow by adding a machine learning backend SDK to Label Studio. You can use `CLIPImageCLassifierAPI.py` as the custom backend server. Follow [this guide](https://labelstud.io/guide/ml.html#How-to-set-up-machine-learning-with-Label-Studio) to set up this server connection.

With this backend server you can perform 2 tasks:
- Dynamically pre-annotate data based on model inference results
- Retrain or fine-tune a model based on recently annotated data

You can use `commands.txt` to start your custom backend.

## Dataset

The fashion challenge dataset can be found
[on Sharepoint](https://machinelearningcompany.sharepoint.com/:f:/r/sites/general/Gedeelde%20%20documenten/Public%20Datasets/dataset%20fashion%20challenge?csf=1&web=1&e=01I4zU "Fashion challenge dataset").

## License

This project is licensed under the terms of the MIT license.

## Author

© Roel Duijsings
