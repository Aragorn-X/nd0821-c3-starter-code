# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import os
import csv
import string

# Add code to load in the data.
cur_dir = os.path.abspath(os.curdir)
os.chdir("..")
data_dir = 'data'
csv_file = 'census_clean.csv'
data_path = os.path.join(os.path.abspath(os.curdir), data_dir, csv_file)
data = pd.read_csv(data_path)
input('wait')
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.

# Train and save a model.
