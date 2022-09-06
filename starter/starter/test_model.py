import os
import logging
import pandas as pd
#import train_model
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

# from ml import data, model

from train_model import import_data, training_model, save_model
from sklearn.model_selection import train_test_split


logging.basicConfig(
    filename='logs/model_testing.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb



def test_import(import_data):
    '''
    test data import:
    - check if file exists in directory
    '''
    try:
        import_data("../data/census_clean.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err


def test_training(training_model):
    '''
    test training_model:
    - verify if the training_model output has 'fit' method
    '''
    # cur_dir = os.path.abspath(os.curdir)
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

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    output_model = training_model(X_train, y_train)

    try:
        assert hasattr(output_model, 'fit')
        logging.info("Testing training_model: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing train_models: The model doesn't appear to have the 'fit' method")
        raise err


def test_saving(save_model):
    cur_dir = os.path.abspath(os.curdir)
    data_dir = 'data'
    csv_file = 'census_clean.csv'  # cleaned file (all spaces removed)
    data_path = os.path.join(cur_dir, data_dir, csv_file)
    data_df = import_data(data_path)
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

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    output_model = training_model(X_train, y_train)

    model_dir = 'model'
    model_path = os.path.join(os.path.abspath(os.curdir), model_dir, 'model.pkl')

    try:
        save_model(model_path, output_model)
        logging.info("Testing save_model: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing save_model: The file wasn't found")
        raise err


if __name__ == "__main__":
    test_import(import_data)
    test_training(training_model)
    test_saving(save_model)
