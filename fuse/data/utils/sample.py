from typing import Dict, Hashable
from fuse.utils.ndict import NDict

"""
helper utilities for creating empty samples, and setting and getting sample_id within samples

A sample is a NDict, which is a special "flavor" of a dictionry, allowing accessing elements within it using x['a.b.c.d'] instead of x['a']['b']['c']['d'],
which is very useful as it allows defining a nested element, or a nested sub-dict using a single string.

The bare minimum that a sample is required to contain are:

'initial_sample_id' - this is an arbitrary (Hashable) identifier. Usually a string, but doesn't have to be.
    It represnts the initial sample_id that was provided before a pipeline was used to process the sample, and potentially use "sample morphing".
    "sample morphing" means that a sample might change during the pipeline execution.
    1. Discard - one type of morphing is that a sample is being discarded. Example use case is discarding an MRI volume because it has too little segmentation info that interests a certain research design.
    2. Split - another type of morphing is that a sample can be split into multiple samples.
    For example, the initial_sample_id represents an entire CT volume, which results in multiple samples, each having the same initial_sample_id, but a different sample_id,
        each representing a slice within the CT volume which contains enough segmentation information

'sample_id' - the sample id, uniquely identifying it. It must be Hashable. Again, usually a string, but doesn't have to be.

"""


def create_initial_sample(initial_sample_id: Hashable, sample_id=None):
    """
    creates an empty sample dict and sets both sample_id and initial_sample_id
    :param sample_id:
    :param initial_sample_id: optional. If not provided, sample_id will be used for it as well
    """
    ans = NDict()

    if sample_id is None:
        sample_id = initial_sample_id

    set_initial_sample_id(ans, initial_sample_id)
    set_sample_id(ans, sample_id)

    return ans


##### sample_id


def get_sample_id_key() -> str:
    """
    return sample id key
    """
    return "data.sample_id"


def get_sample_id(sample: Dict) -> Hashable:
    """
    extracts sample_id from the sample dict
    """
    if get_sample_id_key() not in sample:
        raise Exception
    return sample[get_sample_id_key()]


def set_sample_id(sample: Dict, sample_id: Hashable):
    """
    sets sample_id in an existing sample dict
    """
    sample[get_sample_id_key()] = sample_id


#### dealing with initial sample id - this is related to morphing, and describes the original provided sample_id, prior to the morphing effect


def get_initial_sample_id_key() -> str:
    """
    return initial sample id key
    """
    return "data.initial_sample_id"


def set_initial_sample_id(sample: Dict, initial_sample_id: Hashable):
    """
    sets initial_sample_id in an existing sample dict
    """
    sample[get_initial_sample_id_key()] = initial_sample_id


def get_initial_sample_id(sample: Dict) -> Hashable:
    """
    extracts initial_sample_id from the sample dict
    """
    if get_initial_sample_id_key() not in sample:
        raise Exception
    return sample[get_initial_sample_id_key()]


####


def get_specific_sample_from_potentially_morphed(sample, sample_id):
    if isinstance(sample, dict):
        assert get_sample_id(sample) == sample_id
        return sample
    elif isinstance(sample, list):
        for curr_sample in sample:
            if get_sample_id(curr_sample) == sample_id:
                return curr_sample
        raise Exception(f"Could not find requested sample_id={sample_id}")
    else:
        raise Exception(
            "Expected the sample to be either a dict or a list of dicts. None does not make sense in this context."
        )

    assert False  # should never reach here
