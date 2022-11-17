import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from evalutils import SegmentationAlgorithm
from evalutils.validators import (UniqueImagesValidator,
                                  UniquePathIndicesValidator)

from fuseimg.datasets.picai_test import PICAI
from picai_baseline.unet.training_setup.default_hyperparam import \
    get_default_hyperparams
from picai_baseline.unet.training_setup.neural_network_selector import \
    neural_network_for_run
from fuse.data.utils.collates import CollateDefault
from picai_baseline.unet.training_setup.preprocess_utils import z_score_norm
from picai_prep.data_utils import atomic_image_write
from picai_prep.preprocessing import Sample, PreprocessingSettings, crop_or_pad, resample_img
from report_guided_annotation import extract_lesion_candidates
from scipy.ndimage import gaussian_filter
from runner import create_model
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning import Trainer
from fuse.dl.lightning.pl_module import LightningModuleDefault

class csPCaAlgorithm(SegmentationAlgorithm):
    """
    Wrapper to deploy trained baseline U-Net model from
    https://github.com/DIAGNijmegen/picai_baseline as a
    grand-challenge.org algorithm.
    """

    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

        # set expected i/o paths in gc env (image i/p, algorithms, prediction o/p)
        # see grand-challenge.org/algorithms/interfaces/ for expected path per i/o interface
        # note: these are fixed paths that should not be modified

        # directory to model weights
        self.algorithm_weights_dir = Path("/opt/algorithm/weights/")

        # path to image files
        self.image_input_dirs = [
            "/input/images/transverse-t2-prostate-mri/",
            # "/input/images/transverse-adc-prostate-mri/",
            # "/input/images/transverse-hbv-prostate-mri/",
            # "/input/images/coronal-t2-prostate-mri/",  # not used in this algorithm
            # "/input/images/sagittal-t2-prostate-mri/"  # not used in this algorithm
        ]
        self.image_input_paths = [list(Path(x).glob("*.mha"))[0] for x in self.image_input_dirs]

        # load clinical information
        with open("/input/clinical-information-prostate-mri.json") as fp:
            self.clinical_info = json.load(fp)

        # path to output files
        self.detection_map_output_path = Path("/output/images/cspca-detection-map/cspca_detection_map.mha")
        self.case_level_likelihood_output_file = Path("/output/cspca-case-level-likelihood.json")

        # create output directory
        self.detection_map_output_path.parent.mkdir(parents=True, exist_ok=True)

        # define compute used for training/inference ('cpu' or 'cuda')
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # extract available clinical metadata (not used for this example)
        self.age = self.clinical_info["patient_age"]

        if "PSA_report" in self.clinical_info:
            self.psa = self.clinical_info["PSA_report"]
        else:
            self.psa = None  # value is missing, if not reported

        if "PSAD_report" in self.clinical_info:
            self.psad = self.clinical_info["PSAD_report"]
        else:
            self.psad = None  # value is missing, if not reported

        if "prostate_volume_report" in self.clinical_info:
            self.prostate_volume = self.clinical_info["prostate_volume_report"]
        else:
            self.prostate_volume = None  # value is missing, if not reported

        # extract available acquisition metadata (not used for this example)
        self.scanner_manufacturer = self.clinical_info["scanner_manufacturer"]
        self.scanner_model_name = self.clinical_info["scanner_model_name"]
        self.diffusion_high_bvalue = self.clinical_info["diffusion_high_bvalue"]

        # define input data specs [image shape, spatial res, num channels, num classes]
        self.img_spec = {
            'image_shape': [20, 256, 256],
            'spacing': [3.0, 0.5, 0.5],
            'num_channels': 3,
            'num_classes': 2,
        }

        # load trained algorithm architecture + weights
        self.models = []
        arch_name = ['unet']
        checkpoint_path = self.algorithm_weights_dir / f'fuse_model.pt'

        # skip if model was not trained for this fold
        if not checkpoint_path.exists():
            print("checkpoint not found!")
            return
        # define the model specifications used for initialization at train-time
        # note: if the default hyperparam listed in picai_baseline was used,
        # passing arguments 'image_shape', 'num_channels', 'num_classes' and
        # 'model_type' via function 'get_default_hyperparams' is enough.
        # otherwise arguments 'model_strides' and 'model_features' must also
        # be explicitly passed directly to function 'neural_network_for_run'
        train={}
        train['target'] == 'seg3d'
        model = create_model(train)
        # load python lightning module
        pl_module = LightningModuleDefault.load_from_checkpoint(
            checkpoint_path, model_dir=self.algorithm_weights_dir, model=model, map_location=self.device, strict=True
        )
        # set the prediction keys to extract (the ones used be the evaluation function).
        pl_module.set_predictions_keys(["model.logits.segmentation", "data.gt.classification"])
        print(f"Loading weights from {checkpoint_path}")
        self.models += [model]
        self.pl_module = pl_module
        print("Complete.")
        print("-"*100)   # generate + save predictions, given images
    def predict(self):

        print("Preprocessing Images ...")
        test_dataset = PICAI.dataset(
            sample_ids=self.image_input_paths ,
        )
        ## Create dataloader
        infer_dataloader = DataLoader(
            dataset=test_dataset,
            shuffle=False,
            drop_last=False,
            collate_fn=CollateDefault(),
            num_workers=8,
        )

        # create lightining trainer.
        pl_trainer = Trainer(
            default_root_dir=self.algorithm_weights_dir,
            accelerator=self.device,
            num_sanity_val_steps=-1,
            auto_select_gpus=True,
        )
        # create a trainer instance
        ensemble_output = pl_trainer.predict(self.pl_module , infer_dataloader, return_predictions=True)

        # read and resample images (used for reverting predictions only)
        sitk_img = [
            sitk.ReadImage(str(path)) for path in self.image_input_paths
        ]
        resamp_img = [
            sitk.GetArrayFromImage(
                resample_img(x, out_spacing=self.img_spec['spacing'])
            )
            for x in sitk_img
        ]

        # revert softmax prediction to original t2w - reverse center crop
        cspca_det_map_sitk: sitk.Image = sitk.GetImageFromArray(crop_or_pad(
            ensemble_output, size=resamp_img[0].shape))
        cspca_det_map_sitk.SetSpacing(list(reversed(self.img_spec['spacing'])))

        # revert softmax prediction to original t2w - reverse resampling
        cspca_det_map_sitk = resample_img(cspca_det_map_sitk,
                                          out_spacing=list(reversed(sitk_img[0].GetSpacing())))

        # process softmax prediction to detection map
        cspca_det_map_npy = extract_lesion_candidates(
            sitk.GetArrayFromImage(cspca_det_map_sitk), threshold='dynamic')[0]

        # remove (some) secondary concentric/ring detections
        cspca_det_map_npy[cspca_det_map_npy<(np.max(cspca_det_map_npy)/5)] = 0

        # make sure that expected shape was matched after reverse resampling (can deviate due to rounding errors) 
        cspca_det_map_sitk: sitk.Image = sitk.GetImageFromArray(crop_or_pad(
            cspca_det_map_npy, size=sitk.GetArrayFromImage(sitk_img[0]).shape))

        # works only if the expected shape matches
        cspca_det_map_sitk.CopyInformation(sitk_img[0])

        # save detection map
        atomic_image_write(cspca_det_map_sitk, self.detection_map_output_path)

        # save case-level likelihood
        with open(str(self.case_level_likelihood_output_file), 'w') as f:
            json.dump(float(np.max(cspca_det_map_npy)), f)

if __name__ == "__main__":
    csPCaAlgorithm().predict()