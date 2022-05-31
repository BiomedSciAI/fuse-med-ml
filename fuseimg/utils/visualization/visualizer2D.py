import os
import matplotlib.pyplot as plt
from typing import List
from fuseimg.utils.visualization.imaging_multi_plot import *
from typing import Callable, Dict, Any, List ,Union , Tuple
from fuse.utils.ndict import NDict
from pycocotools import mask as maskUtils
from fuseimg.utils.typing.typed_element import TypedElement
from fuse.data.visualizer.visualizer_base import VisualizerBase


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

def convert_COCO_to_mask(elements : Any,height : int  ,width: int, segmentation_type : str )-> Dict:
    """
    converts COCO type to mask
    :param elements:  input in any COCO format
    :param height: original image height in pixels
    :param width: original image width in pixels
    :param segmentation_type: DataTypeImaging
    :return  output mask 
    """     
    if segmentation_type == "UCRLE":
        elements = [convert_uncompressed_RLE_COCO_type(element,height,width)  for element in elements]    
    elif segmentation_type == "CRLE":
        elements = [convert_compressed_RLE_COCO_type(element,height,width)  for element in elements]     
    return elements
        
class Imaging2dVisualizer(VisualizerBase):
    """
    Creats a matplotlib image that can show a grid of images with segmentations,
    it supports all the segmentation types in DataTypeImaging 
    """
    def __init__(self, cmap :str, format : str ="png", coco_converter : Callable = None) -> None:
        """
        :param cmap : color map for matplotlib 
        :param format : to which file extention to save the image
        :param coco_converter : function that convert the input format to COCO type format
        """
        super().__init__()
        self._cmap = cmap
        self._format = format
        self._coco_converter = coco_converter
    
    def process(self, vis_data: Union[List, NDict]) -> Tuple[List[Dict] , str]:
        """
        get the collected data from the sample, that the visProbe has collected and generated data for actual visualization
        that the _show method can process
        
        :param vis_data: the collected data
        """
        if isinstance(vis_data, NDict):
            vis_data = [vis_data]
        res = []
                    
        for element in vis_data :
            vitem = TypedElement()
            for arg_name, value in element.items() :
                if arg_name in ["height", "width", "name"]:
                    continue
                if self._coco_converter != None :
                    value = self._coco_converter(value, arg_name)
                if arg_name in ["CRLE", "UCRLE"] :
                    element['seg'] = convert_COCO_to_mask(value,element['height'], element['width'], arg_name )
            vitem.image = element['image']
            if 'seg' in element.to_dict().keys():
                vitem.seg = element['seg']
            if 'bboxes' in element.to_dict().keys():
                vitem.bboxes = element['bboxes']
            if 'contours' in element.to_dict().keys():
                vitem.contours = element['contours']
            vitem.metadata = element['name']
            res.append(vitem)
            
        return res     
    def _show(self, vis_data : List):
        """
        see super class
        """
        data = self.process(vis_data)
        show_multiple_images_seg(
            imgs=data,
            cmap=self._cmap )
        plt.show()
    def _save(self, vis_data : List , output_path : str , file_name : str ):
        """
        see super class
        """
        data = self.process(vis_data)
        show_multiple_images_seg(
            imgs=data,
            cmap=self._cmap)
        plt.savefig(os.path.join(output_path,file_name+"."+self._format), format = self._format)
