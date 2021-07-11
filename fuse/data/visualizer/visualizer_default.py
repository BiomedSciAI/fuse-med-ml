"""
(C) Copyright 2021 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on June 30, 2021

"""

import logging
from typing import Optional, Iterable, Any, Tuple

import matplotlib.pyplot as plt

from fuse.data.visualizer.visualizer_base import FuseVisualizerBase
from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict
from fuse.utils.utils_logger import log_object_input_state
from fuse.utils.utils_image_processing import FuseUtilsImageProcessing


class FuseVisualizerDefault(FuseVisualizerBase):
    """
    Visualizer for data including single 2D image with optional mask
    """

    def __init__(self, image_name: str, mask_name: Optional[str] = None,
                 label_name: Optional[str] = None, metadata_names: Iterable[str] = tuple(),
                 pred_name: Optional[str] = None):
        """
        :param image_name: hierarchical key name of the image in batch_dict
        :param mask_name:  hierarchical key name of the mask (gt map) in batch_dict.
                           Optional, won't be displayed if not specified.
        :param label_name: hierarchical key name of the to a global label in batch_dict.
                           Optional, won't be displayed if not specified.
        :param metadata_names: list of hierarchical key name of the metadata - will be printed for every sample
        :param pred_name: hierarchical key name of the prediction in batch_dict.
                           Optional, won't be displayed if not specified.
        """
        # log object input state
        log_object_input_state(self, locals())

        # store input parameters
        self.image_pointer = image_name
        self.mask_name = mask_name
        self.label_name = label_name
        self.metadata_pointers = metadata_names
        self.pred_name = pred_name
        self.matching_function = FuseUtilsImageProcessing.match_img_to_input

    def extract_data(self, sample: dict) -> Tuple[Any, Any, Any, Any, Any]:
        """
        extract required data to visualize from sample
        :param sample: global dict of a sample
        :return: image, mask, label, metadata
        """

        # image
        image = FuseUtilsHierarchicalDict.get(sample, self.image_pointer)

        # mask
        if self.mask_name is not None:
            mask = FuseUtilsHierarchicalDict.get(sample, self.mask_name)
        else:
            mask = None

        # label
        if self.label_name is not None:
            label = FuseUtilsHierarchicalDict.get(sample, self.label_name)
        else:
            label = ''

        # mask
        if self.pred_name is not None:
            pred_mask = FuseUtilsHierarchicalDict.get(sample, self.pred_name)
        else:
            pred_mask = None

        # metadata
        metadata = {metadata_ptr: FuseUtilsHierarchicalDict.get(sample, metadata_ptr) for metadata_ptr in
                    self.metadata_pointers}

        return image, mask, label, metadata, pred_mask

    def visualize(self, sample: dict, block: bool = True) -> None:
        """
        visualize sample
        :param sample: batch_dict - to extract the sample from
        :param block: set to False if the process should not be blocked until the plot will be closed
        :return: None
        """
        # extract data
        image, mask, label, metadata, pred_mask = self.extract_data(sample)

        if mask is not None:
            mask = self.matching_function(mask, image)

        if pred_mask is not None:
            pred_mask = self.matching_function(pred_mask, image)

        # visualize
        num_channels = image.shape[0]

        if pred_mask is not None:
            fig, ax = plt.subplots(num_channels, pred_mask.shape[0]+1, squeeze=False)
        else:
            fig, ax = plt.subplots(num_channels, 1, squeeze=False)

        for channel_idx in range(num_channels):
            ax[channel_idx, 0].title.set_text('image (ch %d) (lbl %s)' % (channel_idx, str(label)))

            ax[channel_idx, 0].imshow(image[channel_idx].squeeze(), cmap='gray')
            if mask is not None:
                ax[channel_idx, 0].imshow(mask[channel_idx], alpha=0.3)

            if pred_mask is not None:
                for c_id in range(pred_mask.shape[0]):
                    max_prob = pred_mask[c_id].max()
                    ax[channel_idx, c_id+1].title.set_text('image (ch %d) (max prob %s)' % (channel_idx, str(max_prob)))

                    ax[channel_idx, c_id+1].imshow(image[channel_idx].squeeze(), cmap='gray')
                    ax[channel_idx, c_id+1].imshow(pred_mask[c_id], alpha=0.3)

        lgr = logging.getLogger('Fuse')
        lgr.info('------------------------------------------')
        lgr.info(metadata)
        lgr.info('image label = ' + str(label))
        lgr.info('------------------------------------------')

        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        fig.tight_layout()
        plt.show(block=block)

    def visualize_aug(self, orig_sample: dict, aug_sample: dict, block: bool = True) -> None:
        """
        Visualise and compare augmented and non-augmented version of the sample
        :param orig_sample: batch_dict to extract the original sample from
        :param aug_sample: batch_dict to extract the augmented sample from
        :param block: set to False if the process should not be blocked until the plot will be closed
        :return: None
        """
        # extract data
        orig_image, orig_mask, orig_label, orig_metadata = self.extract_data(orig_sample)
        aug_image, aug_mask, aug_label, aug_metadata = self.extract_data(aug_sample)

        # visualize
        num_channels = orig_image.shape[0]

        fig, ax = plt.subplots(num_channels, 2, squeeze=False)
        for channel_idx in range(num_channels):
            # orig
            ax[channel_idx, 0].title.set_text('image (ch %d) (lbl %s)' % (channel_idx, str(orig_label)))
            ax[channel_idx, 0].imshow(orig_image[channel_idx].squeeze(), cmap='gray')
            if (orig_mask is not None) and (None not in orig_mask):
                ax[channel_idx, 0].imshow(orig_mask, alpha=0.3)

            # augmented
            ax[channel_idx, 1].title.set_text('image (ch %d) (lbl %s)' % (channel_idx, str(aug_label)))
            ax[channel_idx, 1].imshow(aug_image[channel_idx].squeeze(), cmap='gray')
            if (aug_mask is not None) and (None not in aug_mask):
                ax[channel_idx, 1].imshow(aug_mask, alpha=0.3)
        lgr = logging.getLogger('Fuse')
        lgr.info('------------------------------------------')
        lgr.info("original")
        lgr.info(orig_metadata)
        lgr.info('image label = ' + str(orig_label))
        lgr.info("augmented")
        lgr.info(aug_metadata)
        lgr.info('image label = ' + str(aug_label))
        lgr.info('------------------------------------------')

        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        fig.tight_layout()
        plt.show(block=block)
