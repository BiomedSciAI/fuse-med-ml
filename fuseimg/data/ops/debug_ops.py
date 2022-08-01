import cv2

from fuse.data.ops.op_base import OpBase
from fuseimg.utils.typing.key_types_imaging import DataTypeImaging
from fuseimg.data.ops.ops_common_imaging import OpApplyTypesImaging
from fuse.utils.ndict import NDict

# import SimpleITK as sitk


def no_op(input_tensor):
    return input_tensor


def draw_grid_3d_op(
    input_tensor, start_slice=0, end_slice=None, line_color=255, thickness=10, type_=cv2.LINE_4, pxstep=50
):
    """
    Draws a grid pattern.
    #todo: it is possible to change this function to support both 2d and 3d

    :param input_tensor: a numpy array, either HW  format for grayscale or HWC
    if HWC and C >4 then assumed to be a 3d grayscale

    :param line_color:
    :param thickness:
    :param type_:
    :param pxstep:
    :return:
    """

    # grid = sitk.GridSource(outputPixelType=sitk.sitkUInt16, size=input_tensor.shape, sigma=(0.5, 0.5,0.5), gridSpacing=(100.0, 100.0, 100.0), gridOffset=(0.0, 0.0, 0.0), spacing=(0.2, 0.2, 0.2))
    # grid = sitk.GetArrayFromImage(grid)

    if end_slice is None:
        end_slice = input_tensor.shape[2] - 1

    for s in range(start_slice, end_slice + 1):
        x = pxstep
        y = pxstep
        while x < input_tensor.shape[1]:
            cv2.line(
                input_tensor[..., s],
                (x, 0),
                (x, input_tensor.shape[0]),
                color=line_color,
                lineType=type_,
                thickness=thickness,
            )
            x += pxstep

        while y < input_tensor.shape[0]:
            cv2.line(
                input_tensor[..., s],
                (0, y),
                (input_tensor.shape[1], y),
                color=line_color,
                lineType=type_,
                thickness=thickness,
            )
            y += pxstep

    return input_tensor


# Define function to draw a grid
def draw_grid(im, grid_size):
    # Draw grid lines
    # im = Image.fromarray(im)
    # im = im.astype(np.float32)
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))
    return im


class OpDrawGrid(OpBase):
    """
    draws a 2d grid on the input tensor for debugging
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, sample_dict: NDict, key: str, grid_size):
        img = sample_dict[key]
        draw_grid(img, grid_size=grid_size)

        sample_dict[key] = img
        return sample_dict


op_draw_grid_img = OpApplyTypesImaging({DataTypeImaging.IMAGE: (OpDrawGrid(), {})})
