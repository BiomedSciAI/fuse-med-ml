import unittest

from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuseimg.data.ops.color import OpClip, OpToRange

from fuse.utils.ndict import NDict

import numpy as np


class TestOps(unittest.TestCase):
    
    def test_basic_1(self):
        """
        Test basic imaging ops
        """

        sample = NDict()
        sample["data.input.img"] = np.array([5, 0.5, -5, 3])        

        pipeline = PipelineDefault('test_pipeline', [
            #(op_normalize_against_self, {} ),
            (OpClip(), dict(key="data.input.img", clip=(-0.5, 3.0))),
            (OpToRange(), dict(key="data.input.img", from_range=(-0.5, 3.0), to_range=(-3.5, 3.5))),            
        ])
        
        sample = pipeline(sample)
        
        self.assertLessEqual(sample['data.input.img'].max(), 3.5)
        self.assertGreaterEqual(sample['data.input.img'].min(), -3.5)
        self.assertEqual(sample['data.input.img'][-1], 3.5)

    # FIXME: visualizer        
    # def test_basic_show(self):
    #     """
    #     Test standard backward and forward pipeline
    #     """

    #     sample = TestOps.create_sample_1(views=1)        
    #     visual = Imaging2dVisualizer()
    #     VProbe = partial(VisProbe, 
    #                      keys=  ["data.viewpoint1.img", "data.viewpoint1.seg" ], 
    #                      type_detector=type_detector_imaging,
    #                      visualizer = visual, cache_path="~/")
    #     show_flags = VisFlag.COLLECT | VisFlag.FORWARD | VisFlag.ONLINE

    #     image_downsample_factor = 0.5
    #     pipeline = PipelineDefault('test_pipeline', [
    #         (OpRepeat(OpLoadImage(), [
    #             dict(key_in = 'data.viewpoint1.img_filename', key_out='data.viewpoint1.img'),
    #             dict(key_in = 'data.viewpoint1.seg_filename', key_out='data.viewpoint1.seg')]), {}),
    #         (op_select_slice, {"slice_idx": 50}),
    #         (op_to_int_image_space, {} ),
    #         (op_draw_grid, {"grid_size": 50}),
    #         (VProbe(flags=VisFlag.SHOW_ALL_COLLECTED | VisFlag.FORWARD|VisFlag.REVERSE|VisFlag.ONLINE), {}),
    #         (OpSample(OpAffineTransform2D(do_image_reverse=True)), {
    #             'auto_center' : True,
    #             'output_safety_size_rel': 2.0, #this is only the buffer
    #             'final_scale': image_downsample_factor,
    #             'rotate':  Uniform(-180.0,360.0),    #double range (was middle of range originaly)    #-6.0,12.0],                    #['dist@uniform',-90.0,180.0],           #uniform(-90.0, 180.0),        
    #             'resampling_api': 'cv',
    #             'zoom':  Uniform(1.0,0.5),           #uniform(1.0, 0.1), 1.0,
    #             'translate_rel_pre' : 0.0, #['dist@uniform',0.0,0.05],           #uniform(0.0,0.05),
    #             #'interp' : 'linear', #1 is linear, 0 is nearest - notice - nearest may have a problem in opencv resampling_api
    #             'interp': 'linear', #Choice(['linear','nearest']),
    #             'flip_lr': RandBool(0.5)}),
    #         (OpCropNonEmptyAABB(), {}),
    #         (VProbe( flags=VisFlag.COLLECT | VisFlag.FORWARD | VisFlag.ONLINE), {}),
    #         # (OpSample(op_gamma), dict(gamma=Uniform(0.8,1.2), gain=Uniform(0.9,1.1), clip=(0,1))),
    #     ])
        
    #     sample = pipeline(sample)
    #     rev = pipeline.reverse(sample, key_to_follow='data.viewpoint1.img', key_to_reverse='data.viewpoint1.img')
    



if __name__ == '__main__':
    unittest.main()
    