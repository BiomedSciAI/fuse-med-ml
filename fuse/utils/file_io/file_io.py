import errno
from typing import Iterable, List, Dict, Optional, Tuple, Union, Any
import pickle
import bz2
import gzip
import socket
import os
import numpy as np
import time
import datetime

import pandas as pd
import shutil
import h5py
import hdf5plugin
from scipy.io import arff

from fuse.utils.misc.misc import Misc

###note - it is required that hdf5 support will be installed in a way that blosc is supported as well
# the recommended way is to install it in the following way:
# pip install h5py hdf5plugin
# conda install -y pytables
# and when using, import tables  before everything else


def save_pickle(obj: Any, output_filename: str, compress: bool = False, verbose: int = 0) -> None:
    """
    Pickles object to a file, optionally compressing it.
    Note - when compress=True, gzip will be used, and a ".gz" extension will be adde if not already present.
    returns: the output_filename, potentially with the added extension.
    """
    if compress and not output_filename.endswith(".gz"):
        output_filename += ".gz"
    use_open = gzip.open if (compress or output_filename.endswith(".gz")) else open
    with use_open(output_filename, "wb") as f:
        pickle.dump(obj, f)
    if verbose > 0:
        print("saved pickle: ", output_filename)
    return output_filename


def load_pickle(filename: str) -> Any:
    """
    Loads content from a pickle file
    """
    use_open = open
    if filename.endswith(".gz"):
        use_open = gzip.open
    elif filename.endswith(".pbz2"):
        use_open = bz2.open
    with use_open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle_safe(obj: Any, output_filename: str, compress: bool = False, verbose: bool = 0) -> None:
    """
    a multi-threading/multi-process safe version of save_pickle()
    """

    scrambed_filename = get_randomized_postfix_name(output_filename)

    use_open = gzip.open if compress else open
    if compress and not output_filename.endswith(".gz"):
        output_filename += ".gz"
        scrambed_filename += ".gz"

    with use_open(scrambed_filename, "wb") as f:
        pickle.dump(obj, f)

    os.rename(scrambed_filename, output_filename)
    if verbose > 0:
        print("saved pickle: ", output_filename)
    return output_filename


def save_text_file_safe(filename: str, str_content: str) -> None:
    """
    Saves str_content into a created text file.
    This function is multi-threading and multi-processing safe.
    """

    scrambed_filename = get_randomized_postfix_name(filename)
    save_text_file(scrambed_filename, str_content)
    os.rename(scrambed_filename, filename)
    return filename


def save_text_file(file_path: str, content: str = "") -> None:
    """
    Saves str_content into a created text file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        f.write(content)
        f.close()


def read_simple_float_file(file_path: str) -> float:
    """
    Reads a float string from a file. File is expected to contain only the float number text.
    """
    with open(file_path, "r") as fh:
        # printl('succesfully opened session num file.')
        data = fh.read()
        float_value = float(data)
        return float_value


def read_simple_int_file(file_path: str) -> int:
    """
    Reads an int string from a file. File is expected to contain only the int number text.
    """
    with open(file_path, "r") as fh:
        # printl('succesfully opened session num file.')
        data = fh.read()
        int_value = int(data)
        return int_value


def read_text_file(file_path: str) -> str:
    """
    A simple helper function which loads text from a file
    """
    with open(file_path, "r") as fh:
        # printl('succesfully opened session num file.')
        data = fh.read()
        return data


G_host_name = socket.gethostname()


def get_randomized_postfix_name(filename: str, **additional_rand_state) -> str:
    """
    Returns a filename post-fixed with a random hash
    This is mostly used to create a multi-processing safe save operation.
    For example:
    ...
    desired_name = '/path/to/some/dir/file100.blah'
    scrambed_name = get_randomized_postfix_name(desired_name)
    np.save(scrambled_name, ...)
    os.rename(scrambled_name, desired_name)

    @param filename:
    @return:
    """

    import numpy as np
    import hashlib

    # out_dir = os.path.dirname(filename)
    str_for_hash = filename + str(additional_rand_state)
    str_for_hash += G_host_name + "@" + str(os.getpid()) + str(np.random.random())
    hash_val = hashlib.md5(str_for_hash.encode("utf-8")).hexdigest()

    # scrambed_filename = os.path.join(out_dir, hash_val+'.scrambed')
    scrambed_filename = filename + hash_val + ".scramlbed"
    return scrambed_filename


def read_single_str_line_file(file_path: str) -> str:
    with open(file_path, "r") as fh:
        data = fh.read()
        if data[-1] == "\n":
            data = data[:-1]
        return data


def create_simple_timestamp_file(file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        curr_time = time.time()
        time_stamp_str = datetime.datetime.fromtimestamp(curr_time).strftime("%Y-%m-%d %H:%M:%S")
        f.write(time_stamp_str)
        f.close()


def save_hdf5_safe(filename: str, use_blosc: bool = True, **kwarrays):
    """
    multi-threading and multi-processing safe saving content to hdf5 file
    args:
        use_blosc: uses the blosc compression algorithm
    """
    # first validate the request
    for k, d in kwarrays.items():
        if not isinstance(k, str):
            raise Exception(f"only str keys are supported, instead got {type(k)}")
        if not isinstance(d, np.ndarray):
            raise Exception(f"only np.ndarray data is supported, instead got {type(d)}")
    scrambed_filename = get_randomized_postfix_name(filename)
    # import ipdb;ipdb.set_trace()
    with h5py.File(scrambed_filename, "w") as h5f:
        for k, d in kwarrays.items():
            _use_kwargs = {}
            if use_blosc:
                _use_kwargs = hdf5plugin.Blosc()
            h5f.create_dataset(k, data=d, **_use_kwargs)

    os.rename(scrambed_filename, filename)  # '.' + saved_tensors_format)

    return filename


# TODO: CONSIDER supporting slicing more "organically" - for example, changing this into a class instance,
#      which supports something like: x = load_hdf5('blah'); x['a.b'][:2,10:20:3,...]


def load_hdf5(
    filename: str,
    custom_extract: Optional[Dict[str, Union[int, List, Tuple]]] = None,
) -> dict:
    """

    :param filename:
    :param custom_extraction - OPTIONAL:
        optional arg, by default all keys and data is extracted.

        If you want to extract only part of the stored arrays, and optionally sub-parts of individual arrays as well,
        pass a dict to custom_extract in the following format:
            key: hdf5 dataset name
            data: "index description"

        each "index description" can be:
            if None: returns the entire array data
            if an int: returns data[index, ...]

            if a list (or a tuple) of slice/Ellipsis elements to get a subset, for example:
            index=[slice(10,20,2), slice(None,20), Ellipsis] will extract data[10:20:2, :20, ...]
            usually resulting in much faster load time if it's a small part of the data
            (dependening also on alignment and data storage order)


        Usage example:

        load_hdf5('some_file.hdf5',
            {
                'a.b' : None, #will take all of it
                'a.z.c': [slice(1,10,2), slice(None,20), Ellipsis], #will extract data[1:10:2, :20, ...] from it
            }

        Note: if custom_extraction is provided (not None), then only data from keys found in it will be returned
        In the example above, let's say that there was also a dataset named 'z.z.banana' in the file, it will *not* be returned.
        )
    :return:
    """
    # import ipdb;ipdb.set_trace()

    ans = {}
    h5f = h5py.File(filename, "r")
    for k in h5f.keys():
        if custom_extract is not None:
            if k not in custom_extract:
                continue
            else:
                index = custom_extract[k]
        else:
            index = None

        dset = h5f[k]

        if index is None:
            np_arr = dset[:]
        elif isinstance(index, int):
            np_arr = dset[index, ...]
        else:
            assert isinstance(index, (list, tuple))
            assert len(index) > 0
            assert isinstance(
                index[0], (slice, type(Ellipsis))
            )  # checking just the first, but it should be all of them
            np_arr = dset[tuple(index)]
        ans[k] = np_arr

    return ans


def save_dataframe(df: pd.DataFrame, filename: str, **kwargs) -> None:
    """
    Save dataframe into a file. The file format inferred from filename suffix
    Supported types: "csv", "hd5", "hdf5", "pickle", "pkl", "gz", "xslx", "md"
    :param filename: path to the output file
    """
    file_type = filename.split(".")[-1]

    assert file_type in [
        "csv",
        "hd5",
        "hdf5",
        "hdf",
        "pickle",
        "pkl",
        "gz",
        "xslx",
        "md",
    ], f"file type {file_type} not supported"
    if file_type in ["pickle", "pkl", "gz"]:
        df.to_pickle(filename, **kwargs)
    elif file_type == "csv":
        df.to_csv(filename, **kwargs)
    elif file_type in ["hd5", "hdf5", "hdf"]:
        df.to_hdf(filename, **kwargs)
    elif file_type == "xslx":
        df.to_excel(filename, **kwargs)
    elif file_type == "md":
        df.to_markdown(filename, **kwargs)


def read_dataframe(filename: str) -> pd.DataFrame:
    """
    Read dataframe from a file. The file format inferred from filename suffix
    Supported types: "csv", "hd5", "hdf5", "pickle", "pkl", "gz", "xslx"
    :param filename: path to the output file
    """
    file_type = filename.split(".")[-1]

    assert file_type in [
        "csv",
        "hd5",
        "hdf5",
        "hdf",
        "pickle",
        "pkl",
        "gz",
        "xslx",
    ], f"file type {file_type} not supported"
    if file_type in ["pickle", "pkl", "gz"]:
        df = pd.read_pickle(filename)
    elif file_type == "csv":
        df = pd.read_csv(filename)
    elif file_type in ["hd5", "hdf5", "hdf"]:
        df = pd.read_hdf(filename)
    elif file_type == "xslx":
        df = pd.read_excel(filename)
    elif file_type == "arff":
        data = arff.loadarff(filename)
        df = pd.DataFrame(data[0])

    return df


def delete_directory_tree(del_dir: str) -> bool:
    """
    Will recursively delete del_dir
    :returns: True if was able to delete this dir, False if dir was not found
    Note: if the directory was found but could not delete it, an exception will be raised.
    """
    if not os.path.isdir(del_dir):
        return False
    try:
        shutil.rmtree(del_dir)
    except OSError as e:
        print(f"Error while trying to delete cache: {e.filename} - {e.strerror}.")
        raise

    return True


def create_dir(dir_path: str, error_if_exist: bool = False) -> None:
    """
    Create dir
    :param dir_path: path to dir. either relative or full
    :param error_if_exist: if false will ignore already exist error
    :return: None
    """
    if not os.path.isdir(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError as e:
            if e.errno != errno.EEXIST or error_if_exist:
                raise


def remove_dir_content(dir_path: str, ignore_files: Iterable[str] = tuple(), force_reset: bool = False) -> None:
    """
    Remove the content of dir_path ignoring the files under ignore_files.
    If force_reset is False, prompts the user for approval before the deletion.

    :param dir_path: path to dir. either relative or full
    :param ignore_files: list of files to ignore (don't delete them)
    :param force_reset: when False (default), asks the user for approval when deleting content.
        Else, delete without prompting.
    :return: None
    """
    # if no content - do nothing
    files = os.listdir(dir_path)
    files = [file for file in files if file not in ignore_files]
    num_files = len(files)
    if num_files > 0:
        # prompt the user for approval
        force_remove = (
            True
            if force_reset
            else Misc.query_yes_no(question=f"Folder {dir_path} contains {num_files} files. Delete anyway?")
        )
        # iterate over all the content and delete it
        failure = False
        if force_remove:
            for filename in files:
                file_path = os.path.join(dir_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print("Failed to delete %s. Reason: %s" % (file_path, e))
                    failure = True

        # in case remove wasn't approved or failed to delete some of the files
        if not force_remove or failure:
            msg = f"Folder {dir_path} is already used, please remove it and rerun the program"
            print(msg)
            raise Exception(msg)


def create_or_reset_dir(dir_path: str, ignore_files: Iterable[str] = tuple(), force_reset: bool = False) -> None:
    """
    Create dir or reset it if already exists
    :param dir_path: path to dir. either relative or full
    :param ignore_files: list of files to ignore (don't delete them)
    :return: None
    """
    create_dir(dir_path)
    remove_dir_content(dir_path, ignore_files, force_reset)
