import getpass
import pickle
import gzip
import os

import torch
from deepdiff import DeepDiff

from fuse.data.ops import ops_cast


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


def replace_tensors_with_numpy(d, max_n_dim=2):
    d2 = {}
    for k, v in d.items():

        if isinstance(v, torch.Tensor):
            if len(v.shape) > max_n_dim:
                continue
            v = ops_cast.Cast.to_numpy(v)
            print(k, "tensor => numpy")
        d2[k] = v

    return d2


def compare_dicts(dict_obj, ref_dict_obj):
    deep_diff_config = dict(ignore_nan_inequality=True) #, math_epsilon=0.0001)
    for sample_id, sample_dict in dict_obj.items():
        # keys_to_compare = [s for s in sample_dict.keys() if s.startswith('data.input.patch_annotations.')]
        # for k in keys_to_compare:
        #     v1 = sample_dict[k]
        #     v2 = ref_dict_obj[sample_id].get(k+'_T0')
        #     print(k, v1, v2, type(v1), type(v2))
        #     if v2 is None:
        #         print('k does not exist')
        #     elif not isinstance(v1, tuple):
        #         print("---",k, np.abs(v1-v2))
        #         assert np.abs(v1-v2) < 1e-5

        diff = DeepDiff( ref_dict_obj[sample_id], sample_dict, **deep_diff_config)
        if len(diff) > 0:
            print(sample_id, "has diff")
            # print(f'{sample_id}\n{diff}')
    print("success")
