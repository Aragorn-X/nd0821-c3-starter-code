# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import os
from starter.starter.ml import data, model
import pickle
import csv
import pytest
# Add code to load in the data.

def import_data(path):
    '''
    :param path: path to csv file
    :return: df: pandas dataframe
    '''
    df = pd.read_csv(path)
    return df


def training_model(input_array, target_array):
    '''
    :param input_array: input np.array
    :param target_array: target np.array
    :return: output_model: trained model
    '''
    output_model = model.train_model(input_array, target_array)

    return output_model


def save_model(path, t_model):
    '''
    Saving trained model to specified path dir
    :param path: path to model dir
    :param t_model: trained model
    '''
    with open(path, 'wb') as m_file:
        pickle.dump(t_model, m_file)


def slicing_perfo(path, features_lst, df, enc, binarizer, mdl):
    '''
    Slicing and model performances
    :param path: path to log dir
    :param features_lst: categorical feature list
    :param df: dataframe
    :param enc: encoder
    :param binarizer: binarizer
    :param mdl: trained model
    '''

    file_exists = os.path.isfile(path)
    for feature in features_lst:
        for feature_value in df[feature].unique():
            # slicing dataframe with respect to feature and feature_value
            sliced_df = df[df[feature] == feature_value]
            X_slice, y_slice, _, _ = data.process_data(
                sliced_df,
                categorical_features=features_lst,
                label="salary", training=False,
                encoder=enc, lb=binarizer)
            predictions_slice = model.inference(mdl, X_slice)
            precision_sl, recall_sl, f_beta_sl = model.compute_model_metrics(y_slice, predictions_slice)
            with open(path, 'a') as csvfile:
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


if __name__ == "__main__":
    cur_dir = os.path.abspath(os.curdir)
    os.chdir("..")
    data_dir = 'data'
    csv_file = 'census_clean.csv'  # cleaned file (all spaces removed)
    data_path = os.path.join(os.path.abspath(os.curdir), data_dir, csv_file)
    data_df = import_data(data_path)

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
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

    # Process the test data with the process_data function.
    X_test, y_test, _, _ = data.process_data(
        test, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lb
    )

    # Train model
    trained_model = training_model(X_train, y_train)

    # Inference and performance metrics on the test dataset
    predictions = model.inference(trained_model, X_test)
    precision, recall, f_beta = model.compute_model_metrics(y_test, predictions)
    print('precision: ', precision)
    print('recall: ', recall)
    print('f_beta: ', f_beta)

    # Save model
    model_dir = 'model'
    model_path = os.path.join(os.path.abspath(os.curdir), model_dir, 'model.pkl')
    save_model(model_path, trained_model)

    # Save encoder and binarizer
    binarizer_path = os.path.join(os.path.abspath(os.curdir), model_dir, 'model_lb.pkl')
    pickle.dump(lb, open(binarizer_path, 'wb'))
    encoder_path = os.path.join(os.path.abspath(os.curdir), model_dir, 'model_encoder.pkl')
    pickle.dump(encoder, open(encoder_path, 'wb'))



    # Slicing perfo
    """ 
    with open(model_path, 'rb') as mdl_f:
        loaded_model = pickle.load(mdl_f)
    """
    log_dir = 'logs'
    slice_file = 'slicing_perfo.csv'
    log_path = os.path.join(cur_dir, log_dir, slice_file)
    slicing_perfo(log_path, cat_features, test, encoder, lb, trained_model)
