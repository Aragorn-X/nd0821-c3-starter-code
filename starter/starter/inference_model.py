import os
import numpy as np
import pickle

cur_dir = os.path.abspath(os.curdir)
print(cur_dir)
os.chdir("..")
model_dir = 'model'
model_file = 'model.pkl'
model_path = os.path.join(os.path.abspath(os.curdir), model_dir, model_file)
with open(model_path, 'rb') as mdl_f:
    rf_model = pickle.load(mdl_f)

encoder_file = 'model_encoder.pkl'
encoder_path = os.path.join(os.path.abspath(os.curdir), model_dir, encoder_file)
with open(encoder_path, 'rb') as enc_f:
    rf_encoder = pickle.load(enc_f)

binarizer_file = 'model_lb.pkl'
binarizer_path = os.path.join(os.path.abspath(os.curdir), model_dir, binarizer_file)
with open(binarizer_path, 'rb') as bin_f:
    rf_bin = pickle.load(bin_f)

def load_pkl(pth):
    with open(pth, 'rb') as f:
        pkl_f = pickle.load(f)
    return pkl_f


def execute_inference(data: np.array, cat_features: list):
    # cur_dir = os.path.abspath(os.curdir)
    os.chdir("..")
    model_dir = 'model'
    model_file = 'model.pkl'
    model_path = os.path.join(os.path.abspath(os.curdir), model_dir, model_file)
    rf_model = load_pkl(model_path)
    encoder_file = 'model_encoder.pkl'
    encoder_path = os.path.join(os.path.abspath(os.curdir), model_dir, encoder_file)
    rf_encoder = load_pkl(encoder_path)
    binarizer_file = 'model_lb.pkl'
    binarizer_path = os.path.join(os.path.abspath(os.curdir), model_dir, binarizer_file)
    rf_bin = load_pkl(binarizer_path)
