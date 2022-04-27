import matplotlib
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Type, Union , Tuple
import numpy as np
from fuseimg.utils.visualization.multi_label import *
from fuseimg.utils.visualization.visualizer_base import VisualizerBase

        
class Imaging2dVisualizer(VisualizerBase):
    """
    Curently supports only one group per viewed item
    """
    def __init__(self, cmap) -> None:
        super().__init__()
        self._cmap = cmap

    def _show(self, vis_data : List):
        show_multiple_images_seg(
            imgs=vis_data,
            cmap=self._cmap ,
            unify_size=True)
        if os.getenv('DISPLAY') is None or ''==os.getenv('DISPLAY'): #no monitor (for example, pure SSH session)
            matplotlib.use('Agg')
        plt.show()
    def _save(self, vis_data : List , output_path : str , file_name : str , format="png"):
        show_multiple_images_seg(
            imgs=vis_data,
            cmap=self._cmap,
            unify_size=True)
        plt.savefig(os.path.join(output_path,file_name+"."+format), format = format)
