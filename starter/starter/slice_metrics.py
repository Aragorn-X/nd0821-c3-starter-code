# Script computing model metrics on slices.

from sklearn.model_selection import train_test_split
import pandas as pd
import os
from ml import data, model
import pickle
import csv


def load_pkl(pth):
    with open(pth, 'rb') as f:
        pkl_f = pickle.load(f)
    return pkl_f


def import_data(path):
    '''
    :param path: path to csv file
    :return: df: pandas dataframe
    '''
    df = pd.read_csv(path)
    return df


cur_dir = os.path.abspath(os.curdir)
os.chdir("..")
data_dir = 'data'
csv_file = 'census_clean.csv'  # cleaned file (all spaces removed)
data_path = os.path.join(os.path.abspath(os.curdir), data_dir, csv_file)
data_df = import_data(data_path)

train, test = train_test_split(data_df, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

model_dir = 'model'
model_file = 'model.pkl'
model_path = os.path.join(os.path.abspath(os.curdir), model_dir, model_file)
trained_model = load_pkl(model_path)
encoder_file = 'model_encoder.pkl'
encoder_path = os.path.join(os.path.abspath(os.curdir), model_dir, encoder_file)
encoder = load_pkl(encoder_path)
binarizer_file = 'model_lb.pkl'
binarizer_path = os.path.join(os.path.abspath(os.curdir), model_dir, binarizer_file)
lb = load_pkl(binarizer_path)

log_dir = 'logs'
slice_file = 'slice_output.csv'
slice_txt = 'slice_output.txt'
log_path = os.path.join(cur_dir, log_dir, slice_file)
txt_path = os.path.join(cur_dir, log_dir, slice_txt)

file_exists = os.path.isfile(log_path)

for feature in cat_features:
    for feature_value in test[feature].unique():
        # slicing dataframe with respect to feature and feature_value
        sliced_df = test[test[feature] == feature_value]
        X_slice, y_slice, _, _ = data.process_data(
            sliced_df,
            categorical_features=cat_features,
            label="salary", training=False,
            encoder=encoder, lb=lb)
        predictions_slice = model.inference(trained_model, X_slice)
        precision_sl, recall_sl, f_beta_sl = model.compute_model_metrics(y_slice, predictions_slice)
        with open(log_path, 'a') as csvfile:
            fieldnames = ['feature', 'feature_value', 'precision', 'recall', 'f_beta']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()  # writing header only if file does not exist
                file_exists = True

            writer.writerow({'feature': feature,
                             'feature_value': feature_value,
                             'precision': precision_sl,
                             'recall': recall_sl,
                             'f_beta': f_beta_sl})

        with open(txt_path, 'w') as txt_file:
            with open(log_path, 'r') as in_file:
                [txt_file.write(" ".join(row) + '\n') for row in csv.reader(in_file)]

os.remove(log_path)
