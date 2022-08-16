# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import os
from ml import data, model
import pickle


# Add code to load in the data.
cur_dir = os.path.abspath(os.curdir)
os.chdir("..")
data_dir = 'data'
csv_file = 'census_clean.csv'    # cleaned file (all spaces removed)
data_path = os.path.join(os.path.abspath(os.curdir), data_dir, csv_file)
data_df = pd.read_csv(data_path)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data_df, test_size=0.20)

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


X_train, y_train, encoder, lb = data.process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, _, _ = data.process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train model
trained_model = model.train_model(X_train, y_train)

# Inference and performance metrics
predictions = model.inference(trained_model, X_test)
precision, recall, f_beta = model.compute_model_metrics(y_test, predictions)
print('precision: ', precision)
print('recall: ', recall)
print('f_beta: ', f_beta)

# Save model
model_dir = 'model'
model_path = os.path.join(os.path.abspath(os.curdir), model_dir, 'model.pkl')

with open(model_path,'wb') as model_file:
    pickle.dump(trained_model, model_file)


""" 
# load
with open('model.pkl', 'rb') as f:
    clf2 = pickle.load(f)
"""