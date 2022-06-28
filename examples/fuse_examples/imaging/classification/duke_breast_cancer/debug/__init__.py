import gzip
import os
import pickle


def save_object(obj, filename):
    open_func = gzip.open if filename.endswith(".gz") else open
    filename_tmp = filename+ ".del"
    if os.path.exists(filename_tmp):
        os.remove(filename_tmp)
    with open_func(filename_tmp, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    os.rename(filename_tmp, filename)
    return filename


def load_object(filename):
    open_func = gzip.open if filename.endswith(".gz") else open

    with open_func(filename, 'rb') as myinput:
        try:
            res = pickle.load(myinput)
        except RuntimeError as e:
            print("Failed to read", filename)
            raise e
    return res
