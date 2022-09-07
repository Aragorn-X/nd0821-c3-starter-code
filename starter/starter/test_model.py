import os
import logging
import pandas as pd
import pytest

from starter.starter.ml import data, model
from starter.starter import train_model
from sklearn.model_selection import train_test_split


logging.basicConfig(
    filename='logs/model_testing.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture
def path():
    parent_dir = 'starter'
    data_dir = 'data'
    data_file = 'census_clean.csv'
    data_path = os.path.join(os.path.abspath(os.curdir), parent_dir, data_dir, data_file)
    return data_path
def import_data(path):
    '''
    :param path: path to csv file
    :return: df: pandas dataframe
    '''
    df = pd.read_csv(path)
    return df



def test_import_data(path):
    '''
    test data import:
    - check if file exists in directory
    '''
    try:
        import_data(path)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

@pytest.fixture
def data_segregation(path):
    data_df = import_data(path)
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

    X_train, y_train, _, _ = data.process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    return X_train, y_train


def training_output(data_segregation):
    in_put, target = data_segregation
    output = train_model.training_model(in_put, target)
    return output


def test_training_output(data_segregation):
    '''
    test training_output:
    - verify if the training_model output has 'fit' method
    '''
    try:
        assert hasattr(training_output(data_segregation), 'fit')
        logging.info("Testing training_model: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing train_models: The model doesn't appear to have the 'fit' method")
        raise err

@pytest.fixture
def path_to_model():
    model_dir = 'starter/model'
    model_path = os.path.join(os.path.abspath(os.curdir), model_dir, 'model.pkl')
    return model_path


def saving_model(path, path_to_model):
    in_put, target = data_segregation(path)
    model = train_model.training_model(in_put, target)
    train_model.save_model(path_to_model, model)


def test_saving_model(path_to_model):
    saving_model(path_to_model)
    try:
        os.path.isfile(path_to_model)
        logging.info("Testing save_model: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing save_model: The file wasn't found")
        raise err


""" 
def test_saving(save_model):
    cur_dir = os.path.abspath(os.curdir)
    data_dir = 'data'
    csv_file = 'census_clean.csv'  # cleaned file (all spaces removed)
    data_path = os.path.join(cur_dir, data_dir, csv_file)
    data_df = train_model.import_data(data_path)
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

    output_model = train_model.training_model(X_train, y_train)

    model_dir = 'starter/model'
    model_path = os.path.join(os.path.abspath(os.curdir), model_dir, 'model.pkl')

    try:
        save_model(model_path, output_model)
        logging.info("Testing save_model: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing save_model: The file wasn't found")
        raise err
"""

if __name__ == "__main__":
    test_import_data(train_model.import_data)
    #test_training(train_model.training_model)
    #test_saving(train_model.save_model)
