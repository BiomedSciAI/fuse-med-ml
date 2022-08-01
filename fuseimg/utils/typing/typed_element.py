import numpy as np
from fuse.data.key_types import DataTypeBasic
from fuse.data.patterns import Patterns


class TypedElement:
    """
    encapsulates a single item view with all its overlayed data
    """

    def __init__(self, image=None, seg=None, contours=None, bboxes=None, labels=None, metadata=None) -> None:
        assert isinstance(image, (np.ndarray, type(None)))
        assert isinstance(seg, (np.ndarray, type(None)))
        # assert isinstance(contours, (np.ndarray, type(None)))
        # assert isinstance(bboxes, (np.ndarray, type(None)))
        # assert isinstance(labels, (np.ndarray, type(None)))

        self.image = image
        self.seg = seg
        self.contours = contours
        self.bboxes = bboxes
        self.labels = labels
        self.metadata = metadata


def typedElementFromSample(sample_dict, key_pattern, td):
    patterns = Patterns({key_pattern: True}, False)
    all_keys = [k for k in sample_dict.get_all_keys() if patterns.get_value(k)]

    content = {
        td.get_type(sample_dict, k).value: sample_dict[k]
        for k in all_keys
        if td.get_type(sample_dict, k) != DataTypeBasic.UNKNOWN
    }
    keymap = {td.get_type(sample_dict, k): k for k in all_keys if td.get_type(sample_dict, k) != DataTypeBasic.UNKNOWN}
    elem = TypedElement(**content)
    return elem, keymap


def typedElementToSample(sample_dict, typed_element, keymap):
    for k, v in keymap.items():
        sample_dict[v] = typed_element.__getattribute__(k.value)
    return sample_dict
