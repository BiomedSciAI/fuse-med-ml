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

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from fuse.data.visualizer.visualizer_base import FuseVisualizerBase
from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict


class FuseVisualizerImageAnalysis(FuseVisualizerBase):
    """
    Class for producing analysis of an image
    """

    def __init__(self, image_name: str):
        """
        :param image_name: pointer to an image in batch_dict

        """
        self.image_name = image_name

    def visualize(self, sample: Any, block: bool = True):
        """
                visualize sample
                :param sample: batch_dict - to extract the sample from
                :param block: set to False if the process should not be blocked until the plot will be closed
                :return: None
                """
        # extract data
        image = FuseUtilsHierarchicalDict.get(sample, self.image_name)
        image = image.numpy()
        num_channels = image.shape[0]
        for i in range(num_channels):
            channel_image = image[i, ...]
            if len(channel_image.shape) == 3:
                self.visualize_3dimage(channel_image, title="Image and its Histogram of channel:" + str(i), block=block)
            else:
                assert len(channel_image.shape) == 2
                self.visualise_2dimage(channel_image, title="Image and its Histogram of channel:" + str(i), block=block)

    def visualize_3dimage(self, image: np.array, title: str = "Image and its Histogram", bins=256, block: bool = True) -> None:
        def key_event(e, curr_pos):
            if e.key == "right":
                curr_pos[0] = curr_pos[0] + 1
            elif e.key == "left":
                curr_pos[0] = curr_pos[0] - 1
            else:
                return
            curr_pos[0] = curr_pos[0] % image.shape[0]

            axs[0].cla()
            axs[1].cla()
            axs[0].imshow(image[curr_pos[0]])
            axs[1].hist(image[curr_pos[0]].ravel(), bins=bins, fc='k', ec='k')
            fig.canvas.draw()
            plt.suptitle(title + " at slice:" + str(curr_pos[0]))

        fig, axs = plt.subplots(2)
        position_list = [0]
        fig.canvas.mpl_connect('key_press_event', lambda event: key_event(event, position_list))
        axs[0].imshow(image[0])
        axs[1].hist(image.ravel(), bins=bins, fc='k', ec='k')  # calculating histogram
        plt.suptitle(title + " at slice:" + str(position_list[0]))
        plt.show()

    def visualise_2dimage(self, image: np.array, title: str = "Image and its Histogram", block: bool = True) -> None:
        """
        visualize sample
        :param image: image in the form of np.array
        :param block: set to False if the process should not be blocked until the plot will be closed
        :return: None
        """
        fig = plt.figure()
        fig.add_subplot(221)
        plt.title('image')
        plt.imshow(image)

        fig.add_subplot(222)
        plt.title('histogram')
        plt.hist(image.ravel(), bins=256, fc='k', ec='k')  # calculating histogram

        plt.suptitle(title)
        plt.show(block=block)

    def visualize_aug(self, orig_sample: dict, aug_sample: dict, block: bool = True) -> None:
        """
        Visualise and compare augmented and non-augmented version of the sample
        :param orig_sample: original sample
        :param aug_sample: augmented sample
        :param block: set to False if the process should not be blocked until the plot will be closed
        :return: None
        """
        raise NotImplementedError
