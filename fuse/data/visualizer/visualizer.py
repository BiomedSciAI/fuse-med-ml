import os
import matplotlib.pyplot as plt
from typing import List
from fuseimg.utils.visualization.multi_label import *
from fuseimg.utils.visualization.visualizer_base import VisualizerBase

        
class Imaging2dVisualizer(VisualizerBase):
    """
    Curently supports only one group per viewed item
    """
    def __init__(self, cmap :str, format : str ="png") -> None:
        super().__init__()
        self._cmap = cmap
        self._format = format

    def _show(self, vis_data : List):
        show_multiple_images_seg(
            imgs=vis_data,
            cmap=self._cmap ,
            unify_size=True)
        plt.show()
    def _save(self, vis_data : List , output_path : str , file_name : str ):
        show_multiple_images_seg(
            imgs=vis_data,
            cmap=self._cmap,
            unify_size=True)
        plt.savefig(os.path.join(output_path,file_name+"."+self._format), format = self._format)