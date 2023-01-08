"""
(C) Copyright 2021 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on June 30, 2021

"""


import os
import _pickle as pickle
import gzip


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_pkl(fname):
    if fname.endswith(".pkl"):
        with open(fname, "rb") as f:
            return pickle.load(f)
    elif fname.endswith(".gz"):
        with gzip.open(fname, "rb") as f:
            return pickle.load(f)
    else:
        with open(fname + ".pkl", "rb") as f:
            return pickle.load(f)
