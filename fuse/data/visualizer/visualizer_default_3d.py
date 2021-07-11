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
from skimage.color import gray2rgb
from skimage.segmentation import mark_boundaries

from fuse.data.visualizer.visualizer_base import FuseVisualizerBase
from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict
from fuse.utils.utils_logger import log_object_input_state


class Fuse3DVisualizerDefault(FuseVisualizerBase):
    """
    Visualiser for data including 3D volume with optional local annotations
    """

    def __init__(self, image_name: str, mask_name: Optional[str] = None,
                 label_name: Optional[str] = None, metadata_pointers: Iterable[str] = tuple(),
                 ):
        """
        :param image_name: pointer to an image in batch_dict, image will be in shape (B,C,VOL).
        :param mask_name:  optional, pointer mask (gt map) in batch_dict. If mask location is not part of the batch dict -
                            override the extract_data method
        :param label_name: pointer to a global label in batch_dict
        :param metadata_pointers: list of pointers to metadata - will be printed for every sample

        """
        # log object input state
        log_object_input_state(self, locals())

        # store input parameters
        self.image_name = image_name
        self.mask_name = mask_name
        self.label_name = label_name
        self.metadata_pointers = metadata_pointers

    def extract_data(self, sample: dict) -> Tuple[Any, Any, Any, Any]:
        """
        extract required data to visualize from sample
        :param sample: global dict of a sample
        :return: image, mask, label, metadata
        """

        # image
        image = FuseUtilsHierarchicalDict.get(sample, self.image_name)
        assert len(image.shape) == 4
        image = image.numpy()

        # mask
        if self.mask_name is not None:
            if not isinstance(self.mask_name, list):
                self.mask_name = [self.mask_name]
            masks = [FuseUtilsHierarchicalDict.get(sample, mask_name).numpy() for mask_name in self.mask_name]
        else:
            masks = None

        # label
        if self.label_name is not None:
            label = FuseUtilsHierarchicalDict.get(sample, self.label_name)
        else:
            label = ''

        # metadata
        metadata = {metadata_ptr: FuseUtilsHierarchicalDict.get(sample, metadata_ptr) for metadata_ptr in
                    self.metadata_pointers}

        return image, masks, label, metadata

    def visualize(self, sample: dict, block: bool = True) -> None:
        """
        visualize sample
        :param sample: batch_dict - to extract the sample from
        :param block: set to False if the process should not be blocked until the plot will be closed
        :return: None
        """
        # extract data
        image, masks, label, metadata = self.extract_data(sample)
        # visualize
        chan = 0
        chan_image = image[chan, ...]

        def key_event(e: Any, position_list: Any):  # using left/right key to move between slices
            def on_press(e: Any):  # use mouse click in order to toggle mask/no mask
                'toggle the visible state of the two images'
                if e.button:
                    vis_image = plt_img.get_visible()
                    vis_mask = plt_mask.get_visible()
                    plt_img.set_visible(not vis_image)
                    plt_mask.set_visible(not vis_mask)
                    plt.draw()

            if e.key == "right":
                position_list[0] += 1
            elif e.key == "left":
                position_list[0] -= 1
            elif e.key == "up":
                position_list[1] += 1
            elif e.key == "down":
                position_list[1] -= 1
            else:
                return
            position_list[0] = position_list[0] % image.shape[1]
            position_list[1] = position_list[1] % image.shape[0]
            chan_image = image[position_list[1]]

            ax.cla()
            slice_image = gray2rgb(chan_image[position_list[0]])
            if masks is not None:
                slice_image_with_mask = slice_image
                colors = [(1, 1, 0), (0, 1, 1), (1, 0, 1)]
                for index, mask in enumerate(masks):
                    slice_image_with_mask = mark_boundaries(slice_image_with_mask, mask[position_list[0]].astype(int),
                                                            color=colors[index % len(colors)])

            plt.title(f'Slice {position_list[0]} channel {position_list[1]}')
            plt_img = ax.imshow(slice_image)
            plt_img.set_visible(False)
            if (mask is not None) and (None not in mask):
                plt_mask = ax.imshow(slice_image_with_mask)
                plt_mask.set_visible(True)
            fig.canvas.mpl_connect('button_press_event', on_press)
            fig.canvas.draw()

        fig = plt.figure()
        position_list = [0, 0]
        plt.title(f'Slice {position_list[0]} channel {position_list[1]}')
        fig.canvas.mpl_connect('key_press_event', lambda event: key_event(event, position_list))
        ax = fig.add_subplot(111)
        slice_image = gray2rgb(chan_image[0])
        if masks is not None:
            colors = [(1, 1, 0), (0, 1, 1), (1, 0, 1)]
            for index, mask in enumerate(masks):
                slice_image = mark_boundaries(slice_image, mask[position_list[0]].astype(int), color=colors[index % len(colors)])
        ax.imshow(slice_image)

        lgr = logging.getLogger('Fuse')
        lgr.info('------------------------------------------')
        if metadata is not None:
            if isinstance(metadata, dict):
                lgr.info(FuseUtilsHierarchicalDict.to_string(metadata), {'color': 'magenta'})
            else:
                lgr.info(metadata)

        if label is not None and label != '':
            lgr.info('image label = ' + str(label), {'color': 'magenta'})
        lgr.info('------------------------------------------')

        plt.show()

    def visualize_aug(self, orig_sample: dict, aug_sample: dict, block: bool = True) -> None:
        """
        Visualise and compare augmented and non-augmented version of the sample
        :param orig_sample: batch_dict to extract the original sample from
        :param aug_sample: batch_dict to extract the augmented sample from
        :param block: set to False if the process should not be blocked until the plot will be closed
        :return: None
        """

        # extract data
        orig_image, orig_masks, orig_label, orig_metadata = self.extract_data(orig_sample)
        aug_image, aug_masks, aug_label, aug_metadata = self.extract_data(aug_sample)

        # visualize
        def key_event(e: Any, position_list: Any):  # using left/right key to move between slices
            def on_press(e: Any):  # use mouse click in order to toggle mask/no mask
                'toggle the visible state of the two images'
                if e.button:
                    # Toggle image with no augmentations
                    vis_image = plt_img.get_visible()
                    vis_mask = plt_mask.get_visible()
                    plt_img.set_visible(not vis_image)
                    plt_mask.set_visible(not vis_mask)
                    # Toggle image with augmentations
                    vis_aug_image = plt_aug_img.get_visible()
                    vis_aug_mask = plt_aug_mask.get_visible()
                    plt_aug_img.set_visible(not vis_aug_image)
                    plt_aug_mask.set_visible(not vis_aug_mask)

                    plt.draw()

            if e.key == "right":
                position_list[0] += 1
            elif e.key == "left":
                position_list[0] -= 1
            elif e.key == "up":
                position_list[1] += 1
            elif e.key == "down":
                position_list[1] -= 1
            else:
                return
            position_list[0] = position_list[0] % orig_image.shape[1]
            position_list[1] = position_list[1] % orig_image.shape[0]
            chan_image = orig_image[position_list[1]]
            chan_aug_image = aug_image[position_list[1]]
            # clearing subplots
            axs[0].cla()
            axs[1].cla()
            # creating image without augmentations and with toggling mask
            slice_image = gray2rgb(chan_image[position_list[0]])
            plt_img = axs[0].imshow(slice_image)
            plt_img.set_visible(False)

            if orig_masks is not None:
                slice_image_with_mask = slice_image
                colors = [(1, 1, 0), (0, 1, 1), (1, 0, 1)]
                for index, mask in enumerate(orig_masks):
                    slice_image_with_mask = mark_boundaries(slice_image_with_mask, mask[position_list[0]].astype(int),
                                                            color=colors[index % len(colors)])

                plt_mask = axs[0].imshow(slice_image_with_mask)
                plt_mask.set_visible(True)

            # creating image with augmentations and with toggling mask
            slice_aug_image = gray2rgb(chan_aug_image[position_list[0]])
            plt_aug_img = axs[1].imshow(slice_aug_image)
            plt_aug_img.set_visible(False)
            if aug_masks is not None:
                slice_aug_image_with_mask = slice_aug_image
                colors = [(1, 1, 0), (0, 1, 1), (1, 0, 1)]
                for index, mask in enumerate(aug_masks):
                    slice_aug_image_with_mask = mark_boundaries(slice_aug_image_with_mask, mask[position_list[0]].astype(int),
                                                                color=colors[index % len(colors)])

                plt_aug_mask = axs[1].imshow(slice_aug_image_with_mask)
                plt_aug_mask.set_visible(True)
            # drawing
            axs[0].title.set_text(f"Original - Slice {position_list[0]} channel {position_list[1]}")
            axs[1].title.set_text(f"Augmented - Slice {position_list[0]} channel {position_list[1]}")
            fig.canvas.mpl_connect('button_press_event', on_press)
            fig.canvas.draw()

        fig, axs = plt.subplots(ncols=2)
        position_list = [0, 0]
        chan_image = orig_image[position_list[1]]
        chan_aug_image = aug_image[position_list[1]]

        fig.canvas.mpl_connect('key_press_event', lambda event: key_event(event, position_list))
        slice_image = gray2rgb(chan_image[position_list[0]])
        if orig_masks is not None:
            colors = [(1, 1, 0), (0, 1, 1), (1, 0, 1)]
            for index, mask in enumerate(orig_masks):
                slice_image = mark_boundaries(slice_image, mask[position_list[0]].astype(int),
                                              color=colors[index % len(colors)])

        slice_aug_image = gray2rgb(chan_aug_image[0])
        if aug_masks is not None:
            colors = [(1, 1, 0), (0, 1, 1), (1, 0, 1)]
            for index, mask in enumerate(aug_masks):
                slice_aug_image = mark_boundaries(slice_aug_image, mask[position_list[0]].astype(int),
                                                  color=colors[index % len(colors)])

        axs[0].title.set_text(f"Original - Slice {position_list[0]} channel {position_list[1]}")
        axs[1].title.set_text(f"Augmented - Slice {position_list[0]} channel {position_list[1]}")
        axs[0].imshow(slice_image)
        axs[1].imshow(slice_aug_image)
        plt.show()
