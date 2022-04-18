from fuse.utils.ndict import NDict
import matplotlib
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Type, Union , Tuple
import numpy as np
from fuseimg.utils.visualization.multi_label import *
from fuseimg.utils.typing.key_types_imaging import DataTypeImaging
from fuseimg.utils.typing.typed_element import TypedElement
from fuseimg.utils.visualization.visualizer_base import VisualizerBase

        
class Imaging2dVisualizer(VisualizerBase):
    """
    Curently supports only one group per viewed item
    """
    def __init__(self, cmap ,bbox_colors = 'bgrcmk') -> None:
        super().__init__()
        self._cmap = cmap
        self._bbox_colors = bbox_colors

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
                    vitem.ucrle = value['converted_value']
                elif value['type'] is DataTypeImaging.CRLE:
                    vitem.crle = value['converted_value']
                elif value['type'] is DataTypeImaging.SEG:
                    vitem.seg = value['value']
                vitem.metadata = value['name']
            res.append(vitem)
            
        return res 

    def _show(self, vis_data : List):
        show_multiple_images_seg(
            imgs=vis_data,
            cmap=self._cmap ,
            bbox_colors = self._bbox_colors,
            unify_size=True)
        if os.getenv('DISPLAY') is None or ''==os.getenv('DISPLAY'): #no monitor (for example, pure SSH session)
            matplotlib.use('Agg')
        plt.show()
    def _save(self, vis_data : List , output_path : str , file_name : str , format="png"):
        show_multiple_images_seg(
            imgs=vis_data,
            cmap=self._cmap,
            color = self._bbox_colors,
            unify_size=True)
        plt.savefig(os.path.join(output_path,file_name+"."+format), format = format)
