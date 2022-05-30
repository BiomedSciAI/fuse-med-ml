import numpy as np
from fuse.data.key_types import DataTypeBasic
from fuse.data.patterns import Patterns
from fuse.utils.ndict import NDict
from fuseimg.utils.typing.key_types_imaging import DataTypeImaging

class TypedElement:
    '''
        encapsulates a single item view with all its overlayed data
    '''
    def __init__(self, image=None, seg=None, contours=None, bboxes=None, crle=None, ucrle=None, labels=None, metadata=None) -> None:
        assert isinstance(image, (np.ndarray, type(None)))
        assert isinstance(seg, (np.ndarray, type(None)))
        assert isinstance(contours, (np.ndarray, type(None)))
        assert isinstance(bboxes, (np.ndarray, type(None)))
        assert isinstance(crle, (np.ndarray, type(None)))
        assert isinstance(ucrle, (np.ndarray, type(None)))
        assert isinstance(labels, (np.ndarray, type(None)))

        self.image = image
        self.seg = seg
        self.contours = contours
        self.bboxes = bboxes
        self.crle = crle
        self.ucrle = ucrle
        self.labels = labels
        self.metadata = metadata
