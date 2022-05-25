from abc import ABC, abstractclassmethod, abstractmethod
from typing import Callable, Dict, Any, List ,Union , Tuple
from fuse.utils.ndict import NDict
from torch import Tensor
from torchvision.utils import save_image
import matplotlib
import numpy as np
import nibabel as nib
import os
from pycocotools import mask as maskUtils
from fuseimg.utils.typing.key_types_imaging import DataTypeImaging
from fuseimg.utils.typing.typed_element import TypedElement


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
class VisualizerBase(ABC):
    
    def __init__(self, coco_converter : Callable = None) -> None:
        """
        initialize the visualizer
        :param coco_converter : function that convert the input format to COCO type format
        """
        super().__init__()
        self._coco_converter = coco_converter
        
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
                if self._coco_converter != None :
                    value['value'] = self._coco_converter(value['value'],value['type'])
                if value['type'] is DataTypeImaging.IMAGE:
                    vitem.image = value['value']
                elif value['type'] is DataTypeImaging.BBOX:
                    vitem.bboxes = value['value']
                elif value['type'] is DataTypeImaging.CTR:
                    vitem.contours = value['value']
                elif value['type'] in [DataTypeImaging.CRLE, DataTypeImaging.UCRLE ]:
                    value['value'] = convert_COCO_to_mask(value['value'],value['height'],value['width'],value['type'] )
                    value['type'] = DataTypeImaging.SEG
                if value['type'] is DataTypeImaging.SEG:
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
