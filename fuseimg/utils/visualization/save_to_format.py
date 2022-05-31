from typing import  List
from fuse.utils.ndict import NDict
from torch import Tensor
from fuse.data.visualizer.visualizer_base import VisualizerBase
import nibabel as nib
import os



    
class SaveToFormat(VisualizerBase):
    """
    basic visualizer that saves medical image to known formats, if operated in ONLINE mode it will print the images as bit array
    note: it supports only nifti format at the moment
    """    
    def __init__(self, format = "nii.gz") -> None:
        super().__init__()
        self._format = format
        
    def _show(self, vis_data):
        if isinstance(vis_data, NDict):
            print("showing single item")
            print(vis_data)
        else:
            print(f"comparing {len(vis_data)} items:")
            for item in vis_data:
                print(item)

    def _save(self, vis_data : List , output_path : str , file_name : str ):
        if isinstance(vis_data, NDict):
            vis_data = [vis_data]
        if self._format == "nii.gz" :
            for item in vis_data:
                image = item['image'] 
                if isinstance(image,Tensor) :
                    image = item['image'].numpy()
                image = nib.Nifti1Image(image, affine=None)
                nib.loadsave.save(image,os.path.join(output_path,"img_"+item['name']+"_"+file_name+"."+self._format)  )  
                if "seg" in item.to_dict().keys() :
                    seg = item['seg']
                    if isinstance(seg,Tensor) :
                        seg = seg.numpy()
                    seg = nib.Nifti1Image(seg, affine=None)
                    nib.loadsave.save(seg,os.path.join(output_path,"seg_"+item['name']+"_"+file_name+"."+self._format)  )  
