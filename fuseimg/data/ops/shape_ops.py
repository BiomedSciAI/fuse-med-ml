from typing import Tuple, List
import numpy as np
from torch import Tensor
import skimage
import torch
import torchvision.transforms.functional as TTF

from fuse.utils.ndict import NDict

from fuse.data.ops.op_base import OpBase
from skimage import measure
from fuseimg.utils.typing.key_types_imaging import DataTypeImaging
from fuseimg.data.ops.ops_common_imaging import OpApplyTypesImaging


def sanity_check_HWC(input_tensor):
    if 3 != input_tensor.ndim:
        raise Exception(f"expected 3 dim tensor, instead got {input_tensor.shape}")
    assert input_tensor.shape[2] < input_tensor.shape[0]
    assert input_tensor.shape[2] < input_tensor.shape[1]


def sanity_check_CHW(input_tensor):
    if 3 != input_tensor.ndim:
        raise Exception(f"expected 3 dim tensor, instead got {input_tensor.shape}")
    assert input_tensor.shape[0] < input_tensor.shape[1]
    assert input_tensor.shape[0] < input_tensor.shape[2]


class OpHWCToCHW(OpBase):
    """
    HWC (height, width, channel) to CHW (channel, height, width)
    """

    def __call__(self, sample_dict: NDict, key: str) -> NDict:
        """
        :param key: key to torch tensor of shape [H, W, C]
        """
        input_tensor: Tensor = sample_dict[key]

        sanity_check_HWC(input_tensor)
        input_tensor = input_tensor.permute(dims=(2, 0, 1))
        sanity_check_CHW(input_tensor)

        sample_dict[key] = input_tensor
        return sample_dict


class OpCHWToHWC(OpBase):
    """
    CHW (channel, height, width) to HWC (height, width, channel)
    """

    def __call__(self, sample_dict: NDict, key: str) -> NDict:
        """
        :param key: key to torch tensor of shape [C, H, W]
        """
        input_tensor: Tensor = sample_dict[key]

        sanity_check_CHW(input_tensor)
        input_tensor = input_tensor.permute(dims=(1, 2, 0))
        sanity_check_HWC(input_tensor)

        sample_dict[key] = input_tensor
        return sample_dict


class OpSelectSlice(OpBase):
    """
    select one slice from the input tensor,
    from the first dimension of a >2 dimensional input
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, sample_dict: NDict, key: str, slice_idx: int):
        """
        :param slice_idx: the index of the selected slice from the 1st dimmention of an input tensor
        """

        img = sample_dict[key]
        if len(img.shape) < 3:
            return sample_dict

        img = img[slice_idx]
        sample_dict[key] = img
        return sample_dict


class OpResizeAndPad2D(OpBase):
    """
    Resize and Pad a 2D image
    """

    def __init__(self, number_of_channels: int = 1, pad_value: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.number_of_channels = number_of_channels
        self.pad_value = pad_value

    def pad_image(
        self, inner_image: np.ndarray, padding: Tuple[float, float], resize_to: Tuple[int, int]
    ) -> np.ndarray:
        """
        pads image to requested size ,
        pads both side equally by the same input padding size (left = right = padding[1] , up = down= padding[0] )  ,
        padding default value is zero or minimum value in normalized target range
        :param inner_image: image of shape [H, W, C] of type numpy float32
        :param padding: required padding [x,y]
        :param resize_to: original requested resolution
        :param normalized_target_range: requested normalized image pixels range
        :param number_of_channels: number of color channels in the image
        :return: padded image
        """
        image = self.pad_inner_image(
            inner_image,
            outer_height=resize_to[0] + 2 * padding[0],
            outer_width=resize_to[1] + 2 * padding[1],
            pad_value=self.pad_value,
        )
        return image

    def pad_inner_image(self, image: np.ndarray, outer_height: int, outer_width: int, pad_value: float) -> np.ndarray:
        """
        Pastes input image in the middle of a larger one
        :param image:        image of shape [H, W, C]
        :param outer_height: final outer height
        :param outer_width:  final outer width
        :param pad_value:    value for padding around inner image
        :number_of_channels  final number of channels in the image
        :return:             padded image
        """
        inner_height, inner_width = image.shape[0], image.shape[1]
        h_offset = int((outer_height - inner_height) / 2.0)
        w_offset = int((outer_width - inner_width) / 2.0)
        if self.number_of_channels > 1:
            outer_image = np.ones((outer_height, outer_width, self.number_of_channels), dtype=image.dtype) * pad_value
            outer_image[h_offset : h_offset + inner_height, w_offset : w_offset + inner_width, :] = image
        elif self.number_of_channels == 1:
            outer_image = np.ones((outer_height, outer_width), dtype=image.dtype) * pad_value
            outer_image[h_offset : h_offset + inner_height, w_offset : w_offset + inner_width] = image
        return outer_image

    def resize_image(self, inner_image: np.ndarray, resize_to: Tuple[int, int]) -> np.ndarray:
        """
        resize image to the required resolution
        :param inner_image: image of shape [H, W, C]
        :param resize_to: required resolution [height, width]
        :return: resized image
        """
        inner_image_height, inner_image_width = inner_image.shape[0], inner_image.shape[1]
        if inner_image_height > resize_to[0]:
            h_ratio = resize_to[0] / inner_image_height
        else:
            h_ratio = 1
        if inner_image_width > resize_to[1]:
            w_ratio = resize_to[1] / inner_image_width
        else:
            w_ratio = 1

        resize_ratio = min(h_ratio, w_ratio)
        if resize_ratio != 1:
            inner_image = skimage.transform.resize(
                inner_image,
                output_shape=(int(inner_image_height * resize_ratio), int(inner_image_width * resize_ratio)),
                mode="reflect",
                anti_aliasing=True,
            )
        return inner_image

    def __call__(self, sample_dict: NDict, key: str, resize_to: Tuple, padding: Tuple):
        """
        :param resize_to:               new size of input images, keeping proportions
        :param padding:                 required padding size [x,y]
        """

        img = sample_dict[key]
        # resize
        if resize_to is not None:
            img = self.resize_image(img, resize_to)

        # padding
        if padding is not None:
            img = self.pad_image(img, padding, resize_to)
        sample_dict[key] = img
        return sample_dict


class OpFindBiggestNonEmptyBbox2D(OpBase):
    """
    Finds the the biggest connected component bounding box in the image that is non empty (dark)
    """

    def __init__(self, dark_area_threshold: int = 10, blocks_num: int = 30, **kwargs):
        super().__init__(**kwargs)
        self.dark_area_threshold = dark_area_threshold
        self.blocks_num = blocks_num

    def find_biggest_non_emtpy_bbox(self, img: np.ndarray) -> Tuple[int, int, int, int]:
        """
        split the images into blocks, each block containing (1/30 x 1/30) of the image.
        All blocks above a threshold (10) are considered non-empty.
        Then, the biggest connected component (at the blocks level) is extracted, and its axis-aligned bbox is returned.
        :param img: Image instance , expected 2d integer grayscale image where 0 is black background color
        :param dark_area_threshold: defines grayscale level from which lower is consider dark / outside of body
        :param blocks_num: number of blocks the algorithm split the images into
        :return: four coordinates of the bounding box
        """
        bl_rows = img.shape[0] // self.blocks_num
        bl_cols = img.shape[1] // self.blocks_num

        rows_starts = list(range(self.blocks_num))
        cols_starts = list(range(self.blocks_num))

        rows_starts = list(map(lambda x: x * bl_rows, rows_starts))
        cols_starts = list(map(lambda x: x * bl_cols, cols_starts))

        cells = np.zeros((self.blocks_num, self.blocks_num))

        for ri, r in enumerate(rows_starts):
            for ci, c in enumerate(cols_starts):
                r_end = min(img.shape[0] - 1, r + bl_rows)
                c_end = min(img.shape[1] - 1, c + bl_cols)
                cells[ri, ci] = np.mean(img[r:r_end, c:c_end])

        cells_binary = np.zeros((self.blocks_num, self.blocks_num), dtype=bool)
        cells_binary[cells > self.dark_area_threshold] = True
        blobs_labels = measure.label(cells_binary, background=0)
        regions = measure.regionprops(blobs_labels)

        regions_areas = [r.area for r in regions]
        if len(regions) < 1:
            print("Warning: could not crop properly! fallbacking to full image")
            return 0, 0, img.shape[0] - 1, img.shape[1] - 1

        for i, r in enumerate(regions):
            if r.label == 0:
                regions_areas[i] = -1
                break
        max_ind = np.argmax(regions_areas)
        bbox = regions[max_ind].bbox

        full_img_bbox = [bbox[0] * bl_rows, bbox[1] * bl_cols, (bbox[2] + 1) * bl_rows, (bbox[3] + 1) * bl_cols]
        minr, minc, maxr, maxc = full_img_bbox
        maxr = min(maxr, img.shape[0] - 1)
        maxc = min(maxc, img.shape[1] - 1)

        return minr, minc, maxr, maxc

    def __call__(self, sample_dict: NDict, key: str):
        """ """

        img = sample_dict[key]
        aabb = self.find_biggest_non_emtpy_bbox(img)
        img = img[aabb[0] : aabb[2], aabb[1] : aabb[3]].copy()
        sample_dict[key] = img
        return sample_dict


class OpFlipBrightSideOnLeft2D(OpBase):
    """
    Returns an image where the brigheter half side is on the left, flips the image if the condition does nt hold.
    """

    def __init__(self, max_pixel_value: float = 255.0, dark_region_ratio: float = 15.0, **kwargs):
        super().__init__(**kwargs)
        self.max_pixel_value = max_pixel_value
        self.dark_region_ratio = dark_region_ratio

    def check_bright_side_is_left(self, image: np.ndarray) -> bool:
        """
        checks if the bright side is in the left of the image
        :param image: numpy image , expected 2d grayscale image
        :param max_pixel_value: maximum possible value in the image grayscale format
        :param dark_region_ratio: the raito of possible grayscale values which are considered dark
        :return: True iff the bright side is left
        """
        cols = image.shape[1]
        left_side = image[:, : cols // 2]
        right_side = image[:, cols // 2 :]
        dark_region = self.max_pixel_value / self.dark_region_ratio
        return np.count_nonzero(left_side < dark_region) < np.count_nonzero(right_side < dark_region)

    def __call__(self, sample_dict: NDict, key: str):
        """
        :param image: numpy image , expected 2d grayscale image
        :return: image where the breast is in the left
        """

        image = sample_dict[key]
        if not self.check_bright_side_is_left(image):  # orig
            image = np.fliplr(image)
            sample_dict[key] = image
        return sample_dict


op_select_slice_img_and_seg = OpApplyTypesImaging(
    {DataTypeImaging.IMAGE: (OpSelectSlice(), {}), DataTypeImaging.SEG: (OpSelectSlice(), {})}
)


class OpPad(OpBase):
    """
    Pad the given image on all the sides. Supports Tensor & ndarray.
    """

    def __call__(
        self, sample_dict: NDict, key: str, padding: List[int], fill: int = 0, mode: str = "constant", **kwargs
    ):
        """
        Pad values
        :param key: key to an image in sample_dict - either torch tensor or ndarray
        :param padding: padding on each border. can be differerniate each border by passing a list.
        :param fill: if mode = 'constant', pads with fill's value.
        :param padding_mode: see torch's & numpy's pad functions for more details.
        :param kwargs: numpy's pad function give supports to more arguments. See it's docs for more details.
        """

        img = sample_dict[key]

        if torch.is_tensor(img):
            processed_img = TTF.pad(img, padding, fill, mode)

        elif isinstance(img, np.ndarray):
            # kwargs['constant_values'] = fill
            processed_img = np.pad(img, pad_width=padding, mode=mode, constant_values=fill, **kwargs)

        else:
            raise Exception(f"Error: OpPad expects Tensor or nd.array object, but got {type(img)}.")

        sample_dict[key] = processed_img
        return sample_dict
