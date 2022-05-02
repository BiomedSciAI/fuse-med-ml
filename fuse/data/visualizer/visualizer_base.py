from abc import ABC, abstractclassmethod, abstractmethod
from typing import Dict, Any, List ,Union , Tuple
from fuse.utils.ndict import NDict
from torch import Tensor
from torchvision.utils import save_image
import matplotlib
import numpy as np
import nibabel as nib
import os
from fuseimg.utils.typing.key_types_imaging import DataTypeImaging
from fuseimg.utils.typing.typed_element import TypedElement

class VisualizerBase(ABC):
    
    def __init__(self) -> None:
        super().__init__()
        
    def _preprocess(self, vis_data: Dict[str, Any]):
        """
        get the collected data from the sample, that the visProbe has collected and generated data for actual visualization
        that the _show method can process
        
        :param vis_data: the collected data
        """
        return vis_data

    @abstractmethod
    def _show(self, vis_data: List ):
        """
        actual visualization function, gets a preprocessed collection of items to visualize/compare and shows
        a visualization window that is blocking.
        should be overriden by a specific visualizer
        
        :param vis_data: preprocessed visualization items to display
        """
        raise "should implement abstract method"
    
    @abstractmethod
    def _save(self, vis_data: List , output_path : str , file_name : str):
        """
        actual visualization function, gets a preprocessed collection of items to visualize/compare and saves the visualization to file
        should be overriden by a specific visualizer
        
        :param vis_data: preprocessed visualization items to display
        :param output_path : output folder location
        :param file_name : file name
        """
        raise "should implement abstract method"
    def _preprocess(self, vis_data: Union[List, NDict]) -> Tuple[List[Dict] , str]:
        if isinstance(vis_data, NDict):
            vis_data = [vis_data]
        res = []
        for element in vis_data :
            vitem = TypedElement()
            for item, value in element.items() :
                if value['type'] is DataTypeImaging.IMAGE:
                    vitem.image = value['value']
                elif value['type'] is DataTypeImaging.BBOX:
                    vitem.bboxes = value['value']
                elif value['type'] is DataTypeImaging.CTR:
                    vitem.contours = value['value']
                elif value['type'] is DataTypeImaging.UCRLE:
                    vitem.ucrle = value['value']
                elif value['type'] is DataTypeImaging.CRLE:
                    vitem.crle = value['value']
                elif value['type'] is DataTypeImaging.SEG:
                    vitem.seg = value['value']
                vitem.metadata = value['name']
            res.append(vitem)
            
        return res 
    def show(self, vis_data: List ):
        data  = self._preprocess(vis_data )
        self._show(data )
        
    def save(self,  vis_data: List , op_id: str , output_path : str ):
        data = self._preprocess(vis_data )
        self._save(data , output_path , op_id)

class SaveVisual(VisualizerBase):
    """
    basic visualizer example that just prints the data string representation to the console
    """    
    def __init__(self, format = "nii.gz") -> None:
        super().__init__()
        self._format = format
        
    def _show(self, vis_data):
        if type(vis_data) is dict:
            print("showing single item")
            print(vis_data)
        else:
            print(f"comparing {len(vis_data)} items:")
            for item in vis_data:
                print(item)

    def _save(self, vis_data : List , output_path : str , file_name : str ):
        if self._format == "nii.gz" :
            for item in vis_data:
                if isinstance(item.image,Tensor) :
                    item.image = item.image.numpy()
                    item.image = nib.Nifti1Image(item.image, affine=None)
                    nib.loadsave.save(item.image,os.path.join(output_path,"img_"+item.metadata+"_"+file_name+"."+self._format)  )  
                if item.seg is not None :
                    if isinstance(item.seg,Tensor) :
                        item.seg = item.seg.numpy()
                    item.seg = nib.Nifti1Image(item.seg, affine=None)
                    nib.loadsave.save(item.seg,os.path.join(output_path,"seg_"+item.metadata+"_"+file_name+"."+self._format)  )  