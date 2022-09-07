import os
import numpy as np
import pickle
# from .ml import data, model
from starter.starter.ml import data, model

import pandas as pd


def load_pkl(pth):
    with open(pth, 'rb') as f:
        pkl_f = pickle.load(f)
    return pkl_f


def execute_inference(in_data: pd.DataFrame):
    # cur_dir = os.path.abspath(os.curdir)
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
    model_dir = 'starter/model'
    model_file = 'model.pkl'
    model_path = os.path.join(os.path.abspath(os.curdir), model_dir, model_file)
    rf_model = load_pkl(model_path)
    encoder_file = 'model_encoder.pkl'
    encoder_path = os.path.join(os.path.abspath(os.curdir), model_dir, encoder_file)
    rf_encoder = load_pkl(encoder_path)
    binarizer_file = 'model_lb.pkl'
    binarizer_path = os.path.join(os.path.abspath(os.curdir), model_dir, binarizer_file)
    rf_bin = load_pkl(binarizer_path)
    X, _, _, _ = data.process_data(
        in_data,
        categorical_features=cat_features,
        encoder=rf_encoder,
        lb=rf_bin,
        training=False)
    preds = model.inference(rf_model, X)
    prediction = rf_bin.inverse_transform(preds)[0]
    return prediction

