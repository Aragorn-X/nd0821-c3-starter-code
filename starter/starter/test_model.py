import os
import logging
import pandas as pd
import train_model
from sklearn.model_selection import train_test_split
from ml import data, model


logging.basicConfig(
    filename='logs/model_testing.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


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
    data_df = train_model.import_data(data_path)
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

    X_train, y_train, encoder, lb = data.process_data(
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

    model_dir = 'model'
    model_path = os.path.join(os.path.abspath(os.curdir), model_dir, 'model.pkl')

    try:
        save_model(model_path, output_model)
        logging.info("Testing save_model: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing save_model: The file wasn't found")
        raise err


if __name__ == "__main__":
    test_import(train_model.import_data)
    test_training(train_model.training_model)
    test_saving(train_model.save_model)
