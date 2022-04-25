from typing import List, Optional, Union , Dict , Any
import enum
import numpy as np

from pycocotools import mask as maskUtils
from fuse.utils.ndict import NDict
from fuseimg.utils.visualization.visualizer_base import VisualizerBase
from fuseimg.utils.typing.key_types_imaging import DataTypeImaging
from torch import Tensor
from .op_base import OpBase
from fuse.data.key_types import TypeDetectorBase

class VisFlag(enum.IntFlag):
    COLLECT = 1       #save current state for future comparison        
    SHOW_CURRENT = 2          #show current state                                    
    SHOW_COLLECTED = 4 #show comparison of all previuosly collected states
    CLEAR = 8           #clear all collected states until this point in the pipeline
    ONLINE = 16         #show operations will prompt the user with the releveant plot 
    # REVERSE = 32       #visualization operation will be activated on reverse pipeline execution flow
       

def convert_uncompressed_RLE_COCO_type(element : Dict ,height : int  ,width: int)-> np.ndarray:
    """
    converts uncompressed RLE to COCO default type ( compressed RLE)
    :param element:  input in uncompressed Run Length Encoding (RLE - https://en.wikipedia.org/wiki/Run-length_encoding) 
                    saved in map object example :  {"size": [333, 500],"counts": [26454, 2, 651, 3, 13, 1]}
                    counts first element is how many bits are not the object, then how many bits are the object and so on
    :param height: original image height in pixels
    :param width: original image width in pixels
    :return  output mask 
    """
    p = maskUtils.frPyObjects(element, height, width)
    return p 

def convert_compressed_RLE_COCO_type(element : list ,height : int  ,width: int)-> np.ndarray:
    """
    converts polygon to COCO default type ( compressed RLE)
    :param element:  polygon - array of X,Y coordinates saves as X.Y , example: [[486.34, 239.01, 477.88, 244.78]]
    :param height: original image height in pixels
    :param width: original image width in pixels
    :return   output mask 
    """
    rles = maskUtils.frPyObjects(element, height, width)
    p = maskUtils.merge(rles) 
    p = maskUtils.decode(p)
    return p  



def convert_COCO_to_mask(elements : Any,height : int  ,width: int, segmentation_type : DataTypeImaging )-> Dict:
    """
    converts COCO type to mask
    :param elements:  input in any COCO format
    :param height: original image height in pixels
    :param width: original image width in pixels
    :param segmentation_type: DataTypeImaging
    :return  output mask 
    """     
    if segmentation_type == DataTypeImaging.UCRLE:
        elements = [convert_uncompressed_RLE_COCO_type(element,height,width)  for element in elements]    
    elif segmentation_type == DataTypeImaging.CRLE:
        elements = [convert_compressed_RLE_COCO_type(element,height,width)  for element in elements]     
    return elements
    
class VisProbe(OpBase):
    """
    Handle visualization, saves, shows and compares the sample with respect to the current state inside a pipeline
    In most cases VisProbe can be used regardless of the domain, and the domain specific code will be implemented 
    as a Visualizer inheriting from VisualizerBase. In some cases there might be need to also inherit from VisProbe.

    Important notes:
    - running in a cached environment is dangerous and is prohibited
    - this Operation is not thread safe and so multithreading is also discouraged

    "
    """

    def __init__(self,flags: VisFlag, 
                 keys: Union[List, dict] ,
                 type_detector: TypeDetectorBase,
                 coco_converter = None,
                 name : str ="",
                 sample_id: Union[None, List] = None, 
                 visualizer: VisualizerBase = None,
                 output_path: str = "~/"):
        """ 
          :param flags: operation flags (or possible concatentation of flags using IntFlag), details:   
              COLLECT - save current state for future comparison                                                                                               
              SHOW_CURRENT - show current state                                                                                                                        
              SHOW_COLLECTED - show comparison of all previuosly collected states                                                                              
              CLEAR - clear all collected states until this point in the pipeline                                                                              
              ONLINE - show operations will prompt the user with the releveant plot                                                                        
              OFFLINE - show operations will write to disk (using the caching mechanism) the relevant info (state or states for comparison)                
              FORWARD - visualization operation will be activated on forward pipeline execution flow                                                       
              REVERSE - visualization operation will be activated on reverse pipeline execution flow
          :param keys: for which sample keys to handle visualization, also can be grouped in a dictionary
          :param type_detector : class used to identify objects from the keys
          :param coco_converter : function that convert the input format to COCO type format
          :param name : image name to represnt, if not given a number will be shown.
          :param sample_id: for which sample id's to be activated, if None, active for all samples  
          :param visualizer: the actual visualization handler, depands on domain and use case, should implement Visualizer Base
          :param output_path: root dir to save the visualization outputs in offline mode
          
          few issues to be aware of, detailed in github issues regarding static cached pipeline and multiprocessing
          note - if both forward and reverse are on, then by default, on forward we do collect and on reverse we do show_collected to 
          compare reverse operations
          for each domain we inherit for VisProbe like ImagingVisProbe,...
"""
        super().__init__()
        self._sample_id = sample_id
        self._keys  = keys
        self._flags = flags
        self._coco_converter = coco_converter
        self._name = name
        self._collected_prefix = "data.$vis"
        self._output_path = output_path
        self._visualizer = visualizer
        self._type_detector = type_detector

    def _extract_collected(self, sample_dict: NDict):
        res = []
        if not self._collected_prefix in sample_dict:
            return res
        else:
            for vdata in sample_dict[self._collected_prefix]:
                res.append(vdata)
        return res
    
    def _extract_data(self, sample_dict: NDict, keys : List , name : str):
        res = NDict()
        for key in keys:
            prekey = key.replace(".", "_")
            if isinstance(sample_dict[key] , Tensor ):
                res[f'{prekey}.value'] = sample_dict[key].clone()
            else :
                 res[f'{prekey}.value'] = sample_dict[key].copy()
            if self._coco_converter != None :
                 res[f'{prekey}.value'] = self._coco_converter(res[f'{prekey}.value'])
            res[f'{prekey}.type'] = self._type_detector.get_type(sample_dict, key)
            if res[f'{prekey}.type'] in [DataTypeImaging.CRLE, DataTypeImaging.UCRLE  ]:
                res[f'{prekey}.value'] = convert_COCO_to_mask(res[f'{prekey}.value'],sample_dict['height'],sample_dict['width'],res[f'{prekey}.type'] )
                res[f'{prekey}.type'] = DataTypeImaging.SEG 
            res[f'{prekey}.name'] = name
        return res

    def _handle_flags(self, sample_dict: NDict, op_id: Optional[str]):
        """
        See super class
        """
        # sample was filtered out by its id
        if self._sample_id and self.get_idx(sample_dict) not in self._sample_id:
            return None
       
        vis_data = self._extract_data(sample_dict, self._keys ,self._name)
        name_prefix=""
        if "name" in sample_dict.to_dict().keys():
            name_prefix = sample_dict["name"]
        name_prefix += "."+op_id
        if VisFlag.COLLECT in self._flags:
            if not self._collected_prefix in sample_dict:
                sample_dict[self._collected_prefix] = []
            sample_dict[self._collected_prefix].append(vis_data)

        
        if VisFlag.SHOW_CURRENT in self._flags:
            if VisFlag.ONLINE in self._flags:
                self._visualizer.show(vis_data)
            else :
                self._visualizer.save(vis_data , name_prefix , self._output_path)
        
        if  VisFlag.SHOW_COLLECTED in self._flags:
            vis_data = self._extract_collected(sample_dict) + [vis_data]
            if VisFlag.ONLINE in self._flags:
                self._visualizer.show(vis_data)
            else :
                self._visualizer.save(vis_data , name_prefix, self._output_path)

        if VisFlag.CLEAR in self._flags:
            sample_dict[self._collected_prefix] = []
            
        # TODO - support reverse?
        # if VisFlag.SHOW_COLLECTED in self._flags and VisFlag.REVERSE in self._flags:
        #     sample_dict[self._collected_prefix].pop()

        return sample_dict
        

    def __call__(self, sample_dict: NDict, op_id: Optional[str], **kwargs) -> Union[None, dict, List[dict]]:
        res = self._handle_flags(sample_dict, op_id)
        return res


