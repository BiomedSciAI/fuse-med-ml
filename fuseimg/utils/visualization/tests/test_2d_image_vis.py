
import unittest
from functools import partial
import os
import torch

from fuse.data import PipelineDefault, OpToTensor, OpRepeat
from fuseimg.data.ops.aug.geometry import OpAugAffine2D
from fuseimg.data.ops.image_loader import OpLoadImage , OpDownloadImage
from fuseimg.data.ops.color import  OpToRange, OpToIntImageSpace
from fuseimg.data.ops.aug.color import OpAugColor
from fuseimg.data.ops.debug_ops import OpDrawGrid
from fuse.data.ops.ops_visprobe import VisFlag, VisProbe 
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuseimg.utils.visualization.visualizer import Imaging2dVisualizer

from fuseimg.data.ops.shape_ops import OpSelectSlice

from fuseimg.utils.typing.key_types_imaging import type_detector_imaging
from fuse.data.ops.ops_aug_common import  OpSample
from fuse.data import PipelineDefault, OpSampleAndRepeat, OpToTensor, OpRepeat
from fuse.utils.rand.param_sampler import Uniform , RandInt ,RandBool

from tempfile import gettempdir
from fuseimg.datasets.kits21 import KITS21
import pathlib
import skimage.io as io
from pycocotools.coco import COCO
from fuse.utils.ndict import NDict

def create_sample_1(views=2):
    data_dir = os.path.join(gettempdir(), "kits21_data")
    KITS21.download(data_dir, cases=[100,200])

    sample = NDict()
    if views >= 1:
        sample['data.viewpoint1.img_filename'] = os.path.join(
            data_dir, 'case_00100/imaging.nii.gz') 
        sample['data.viewpoint1.seg_filename'] = os.path.join(
            data_dir, 'case_00100/aggregated_MAJ_seg.nii.gz')
    for view_idx in range(2,views+1):
        sample[f'data.viewpoint{view_idx }.img_filename'] = os.path.join(
            data_dir, 'case_00200/imaging.nii.gz')
        sample[f'data.viewpoint{view_idx }.seg_filename'] = os.path.join(
            data_dir, 'case_00200/aggregated_MAJ_seg.nii.gz')
    
    return sample , data_dir
    
class TestImageVisualizer(unittest.TestCase):
    def test_collect_compare(self):
        """
        Test standard backward and forward pipeline
        """


        sample , data_dir = create_sample_1(views=1)        
        sample["name"] = "kits21_example"
        visual = Imaging2dVisualizer(cmap = 'gray')
        VProbe = partial(VisProbe, 
                         keys=  ["data.viewpoint1.img", "data.viewpoint1.seg" ], 
                         type_detector=type_detector_imaging,
                         visualizer = visual, output_path=os.getcwd())
        repeat_for = [dict(key="data.viewpoint1.img"), dict(key="data.viewpoint1.seg")]
        slice_idx = 190
        pipeline = PipelineDefault('test_pipeline', [
            (OpLoadImage(data_dir), dict(key_in = 'data.viewpoint1.img_filename', key_out='data.viewpoint1.img', format="nib")),
            (OpLoadImage(data_dir), dict(key_in = 'data.viewpoint1.seg_filename', key_out='data.viewpoint1.seg', format="nib")),
            (OpSelectSlice(), dict(key="data.viewpoint1.img", slice_idx = slice_idx)),
            (OpSelectSlice(), dict(key="data.viewpoint1.seg", slice_idx = slice_idx)),
            (OpToIntImageSpace(), dict(key="data.viewpoint1.img") ),
            (OpRepeat(OpToTensor(), kwargs_per_step_to_add=repeat_for), dict(dtype=torch.float32)),
            (VProbe( VisFlag.COLLECT , name = "first"), {}),
            (OpToRange(), dict(key="data.viewpoint1.img", from_range=(-500, 500), to_range=(0, 1))),
            (VProbe( VisFlag.COLLECT , name = "second"), {}),
            (OpSampleAndRepeat(OpAugAffine2D (), kwargs_per_step_to_add=repeat_for), dict(
                rotate=30.0
            )),
            (VProbe( VisFlag.COLLECT, name = "last"), {}),
            (OpSampleAndRepeat(OpAugAffine2D (), kwargs_per_step_to_add=repeat_for), dict(
                rotate=30.0
            )),
            (VProbe( flags=VisFlag.SHOW_COLLECTED ), {}),
            
        ])
        
        sample = pipeline(sample)
        
        self.assertLessEqual(sample['data.viewpoint1.img'].max(), 1.0)
        self.assertGreaterEqual(sample['data.viewpoint1.img'].max().min(), -1.0)
        
    def test_other_seg(self):
        visual = Imaging2dVisualizer(cmap = 'gray' )
        dir_path = pathlib.Path(__file__).parent.resolve()
        annotation_path =os.path.join(dir_path,'inputs/detection/example_coco_new.json')
        cocoGt=COCO(annotation_path)
        #initialize COCO detections api
        resFile=os.path.join(dir_path,'inputs/detection/instances_val2014_fakesegm100_results.json')
        coco=cocoGt.loadRes(resFile)
        catNms=['person','car']
        seg_type_name_map = {'ctr':'segmentation','bbox':'bbox'}
        catIds = coco.getCatIds(catNms)
        imgIds = coco.getImgIds(catIds=catIds )
        for img_id in imgIds:
            for img in coco.loadImgs(ids = [img_id]):
                for segtype in seg_type_name_map.keys() :
                    VProbe = partial(VisProbe, 
                    keys=  ["data.viewpoint1.img" , "data.viewpoint1."+segtype], 
                    type_detector=type_detector_imaging,
                    visualizer = visual, output_path=os.getcwd())
                    pipeline = PipelineDefault('test_pipeline', [
                            (OpDownloadImage(), dict(key_in ='data.viewpoint1.img_filename', key_out='data.viewpoint1.img')),
                            (VProbe( flags=VisFlag.SHOW_COLLECTED ), {}),
                        
                        ])
                    sample_dict = NDict()
                    sample_dict['data.viewpoint1.img_filename'] = "http://images.cocodataset.org/val2014/"+img['file_name'] 
                    sample_dict["height"] = img['height']
                    sample_dict["width"] = img['width']
                    sample_dict["name"] = segtype+"_"+img['file_name'] 
                    target_annIds = cocoGt.getAnnIds(imgIds=img_id, catIds=[str(id) for id in catIds], iscrowd=None)
                    segmentations = []
                    for seg in cocoGt.loadAnns(target_annIds) :
                        segmentations.append(seg[seg_type_name_map[segtype]])
                    sample_dict["data.viewpoint1."+segtype] = segmentations
                    sample_dict = pipeline(sample_dict)
    def tearDown(self) -> None:
        return super().tearDown()


if __name__ == '__main__':
    unittest.main()
