
import os
import _pickle as pickle
import gzip

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_pkl(fname):
    if fname.endswith('.pkl'):
        with open(fname, 'rb') as f:
            return pickle.load(f)
    elif fname.endswith('.gz'):
        with gzip.open(fname, 'rb') as f:
            return pickle.load(f)
    else:
        with open(fname + '.pkl', 'rb') as f:
            return pickle.load(f)
