import getpass
import pickle
import gzip
import os

def get_duke_user_dir():
    return f'/projects/msieve_dev3/usr/{getpass.getuser()}/fuse_examples/duke'

def get_duke_radiomics_user_dir():
    return f'/projects/msieve_dev3/usr/{getpass.getuser()}/fuse_examples/duke_radiomics'

def get_duke_lesion_properties_user_dir():
    return f'/projects/msieve_dev3/usr/{getpass.getuser()}/fuse_examples/duke_lesion_properties'

def ask_user(yes_no_question):
    res = ''
    while res not in ['y', 'n']:
        res = input(f'{yes_no_question}? [y/n]')
    return res =='y'

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
