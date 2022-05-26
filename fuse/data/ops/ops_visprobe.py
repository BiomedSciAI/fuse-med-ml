from typing import List, Optional, Union , Dict , Any , Callable
import enum
import numpy as np
from copy import deepcopy

from fuse.utils.ndict import NDict
from fuse.data.visualizer.visualizer_base import VisualizerBase
from fuseimg.utils.typing.key_types_imaging import DataTypeImaging
from torch import Tensor
from .op_base import OpReversibleBase
from fuse.data.key_types import TypeDetectorBase
from fuse.data.utils.sample import get_sample_id
class VisFlag(enum.IntFlag):
    COLLECT = 1                     #save current state for future comparison        
    VISUALIZE_CURRENT = 2           #show current state                                    
    VISUALIZE_COLLECTED = 4          #show comparison of all previuosly collected states
    CLEAR = 8                       #clear all collected states until this point in the pipeline
    ONLINE = 16                     #show operations will prompt the user with the releveant plot 
    # REVERSE = 32                  #visualization operation will be activated on reverse pipeline execution flow
       

    
class VisProbe(OpReversibleBase):
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
                 visualizer: VisualizerBase,
                 name : str ="",
                 output_path: str = "~/"):
        """ 
          :param flags: operation flags (or possible concatentation of flags using IntFlag), details:   
              COLLECT - save current state for future comparison                                                                                               
              VISUALIZE_CURRENT - visualize current state                                                                                                                        
              VISUALIZE_COLLECTED - visualize comparison of all previuosly collected states                                                                              
              CLEAR - clear all collected states until this point in the pipeline                                                                              
              ONLINE - visualize operations will prompt the user with the releveant plot                                                                        
              OFFLINE - visualize operations will write to disk (using the caching mechanism) the relevant info (state or states for comparison)                
              FORWARD - visualization operation will be activated on forward pipeline execution flow                                                       
              REVERSE - visualization operation will be activated on reverse pipeline execution flow
          :param keys: for which sample keys to handle visualization, also can be grouped in a dictionary
          :param type_detector : class used to identify objects from the keys
          :param coco_converter : function that convert the input format to COCO type format
          :param visualizer: the actual visualization handler, depands on domain and use case, should implement Visualizer Base
          :param name : image name to represnt, if not given a number will be shown.
          :param output_path: root dir to save the visualization outputs in offline mode
          
          few issues to be aware of, detailed in github issues regarding static cached pipeline and multiprocessing
          note - if both forward and reverse are on, then by default, on forward we do collect and on reverse we do show_collected to 
          compare reverse operations
          for each domain we inherit for VisProbe like ImagingVisProbe,...
"""
        super().__init__()
        self._keys  = keys
        self._flags = flags
        self._collected_prefix = "data.$vis"
        self._name = name
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
            res[f'{prekey}.value'] =  deepcopy(sample_dict[key])
            res[f'{prekey}.type'] = self._type_detector.get_type(sample_dict, key)
            if 'height' in sample_dict.keys() and 'width' in sample_dict.keys() :
                res[f'{prekey}.height'] = sample_dict['height']
                res[f'{prekey}.width'] = sample_dict['width']
            res[f'{prekey}.name'] = name
        return res

    def _handle_flags(self, sample_dict: NDict, op_id: Optional[str]):
        """
        See super class
        """
        vis_data = self._extract_data(sample_dict, self._keys ,self._name)
        name_prefix = op_id
        if VisFlag.COLLECT in self._flags:
            if not self._collected_prefix in sample_dict:
                sample_dict[self._collected_prefix] = []
            sample_dict[self._collected_prefix].append(vis_data)

        
        if VisFlag.VISUALIZE_CURRENT in self._flags:
            if VisFlag.ONLINE in self._flags:
                self._visualizer.show(vis_data)
            else :
                self._visualizer.save(vis_data , name_prefix , self._output_path)
        
        if  VisFlag.VISUALIZE_COLLECTED in self._flags:
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

    def reverse(self, sample_dict: NDict, op_id: Optional[str], key_to_reverse: str, key_to_follow: str) -> dict:
        """
        See super class - not implemented for now
        """
        # res = self._handle_flags(VisFlag.REVERSE, sample_dict, op_id)
        # if res is None:
        #     res = sample_dict
        # return res
        return sample_dict

