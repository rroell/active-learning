# © Roel Duijsings
"""
This script reads the 'fashion challenge' dataset from an Excel file, preprocesses the data, and saves it into a CSV file.

The preprocessing involves selecting specific columns to keep, modifying the image paths to include the directory prefix, 
dropping all rows where 'product_sub_category' is 'N/D', and adding four new columns: 'annotation', 'score', 
'prediction', and 'top5'. These new columns are initialized as None and are meant to be populated in later steps.

After preprocessing, the resulting DataFrame is saved as a CSV file.

Global Variables
----------------
FILE_NAME : str
    The name of the CSV file where the resulting DataFrame is saved.
columns_to_keep : list
    The list of columns from the original dataset to keep during preprocessing.
"""
import pandas as pd

FILE_NAME = 'fashion_data_no_ND.csv'

columns_to_keep = ['product_id', 'product_family', 'product_category', 'product_sub_category', 'product_image_path'] 
products = pd.read_excel(r"dataset fashion challenge\text\products.xlsx", usecols=columns_to_keep)

# TEMP: drop category N/D (not decided)
products = products[products['product_sub_category'] != 'N/D']

prefix = r"dataset fashion challenge\images\\"
products['product_image_path'] = prefix + products['product_image_path'].astype(str)

products['annotation'] = None
products['score'] = None
products['prediction'] = None
products['top5'] = None
products.to_csv(FILE_NAME, index=False)

print("Created dataset of size", products.shape, "as", FILE_NAME)