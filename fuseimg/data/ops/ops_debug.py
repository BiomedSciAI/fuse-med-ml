import os
from typing import Callable, Optional, Tuple
from fuse.data.ops.ops_cast import Cast
from fuse.data.utils.sample import get_sample_id
from fuse.utils import NDict
from fuse.utils.file_io.file_io import create_dir
from fuse.data.ops.ops_debug import OpDebugBase
import numpy
import torch
import matplotlib.pyplot as plt


class OpVis2DImage(OpDebugBase):
    """
    Visualize a 2D image (either display it or save it as an image).
    Supports both rgb and gray scale and supports few image formats "channels_first"/"no_channels"/"channels_last"
    It's recommended, but not a must, to run it in a single process.
    Add at the top your script to force single process:
    ```
    from fuse.utils.utils_debug import FuseDebug
    FuseDebug("debug")
    ```
    Example:
    # Display RGB image [3, H, W] already in range [0-256]
    (OpVis2DImage(num_samples=1), dict(key="data.input.img", dtype="int"))

    """

    def __init__(
        self,
        show: bool = True,
        path: str = ".",
        image_format: str = "channels_first",
        image_process_func: Optional[Callable] = None,
        **kwargs,
    ):
        """
        :param show: if set to true will display it, otherwise will save it in `path`
        :param path: location to save the images. Used when `show` set to False
        :param image_format: the format of the image: "channels_first"/"no_channels"/"channels_last"
        :param image_processing_func: callable that get the image (numpy array) and can process it. Typically use to move the range to values matplotlib can display.
        :param kwargs: see super class arguments
        """
        super().__init__(**kwargs)
        self._path = path
        self._image_format = image_format
        self._image_process_func = image_process_func
        self._show = show

    def call_debug(
        self,
        sample_dict: NDict,
        key: str,
        clip: Optional[Tuple] = None,
        dtype: Optional[str] = None,
        figure_kwargs: dict = {},
        **imshow_kwargs,
    ) -> None:
        """
        :param key: sample_dict key to a 2D image. Either tensor or numpy array.
        :param clip: tuple of arguments to numpy.clip if the operation required
        :param dtype: in case the range is [0-256], matplotlib expects int. Set to "int" if this is the case.
        :param figure_kwargs: parameters to pass to plt.figure
        :params imshow_kwargs: extra parameters to pass to plt.imshow
        """

        img = sample_dict[key]
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        assert isinstance(img, numpy.ndarray)

        if clip is not None:
            img = img.clip(*clip)

        if dtype is not None:
            img = Cast.to_numpy(img, dtype=dtype)

        if self._image_format == "channels_first":
            img = numpy.moveaxis(img, 0, -1)
        elif self._image_format == "no_channels":
            img = numpy.expand_dims(img, axis=-1)

        if self._image_process_func is not None:
            img = self._image_process_func(img)

        if "cmap" not in imshow_kwargs:
            if img.shape[-1] == 1:  # assume and force gray scale
                imshow_kwargs["cmap"] = "gray"

        plt.figure(**figure_kwargs)
        plt.imshow(img, **imshow_kwargs)
        if self._show:
            plt.show()
        else:
            if self._name is None:
                filename = os.path.join(self._path, get_sample_id(sample_dict).replace(".", "__")) + ".png"
            else:
                filename = os.path.join(self._path, self._name, get_sample_id(sample_dict).replace(".", "__")) + ".png"
            create_dir(os.path.dirname(filename))
            plt.savefig(filename)  # save fig


class OpVisImageHist(OpDebugBase):
    """
    Visualize single image histogram (either display it or save it as an image).
    ```
    from fuse.utils.utils_debug import FuseDebug
    FuseDebug("debug")
    ```
    Example:
    ```
    (OpVisImageHist(first_sample_only=True), dict(key="data.input.img"))
    ```
    """

    def __init__(self, show: bool = True, path: str = ".", **kwargs):
        """
        :param show: if set to true will display it, otherwise will save it in `path`
        :param path: location to save the images. Used when `show` set to False
        :param kwargs: see super class arguments
        """
        super().__init__(**kwargs)
        self._path = path
        self._show = show

    def call_debug(
        self, sample_dict: NDict, key: str, bins: int = 10, figure_kwargs: dict = {}, **hist_kwargs
    ) -> NDict:
        """
        :param key: sample_dict key to a 2D image. Either tensor or numpy array.
        :param figure_kwargs: parameters to pass to plt.figure
        :params hist_kwargs: extra parameters to pass to plt.hist
        """

        img = sample_dict[key]
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        assert isinstance(img, numpy.ndarray)

        if self._image_format == "channels_first":
            img = numpy.moveaxis(img, 0, -1)
        if self._image_process_func is not None:
            img = self._image_process_func(img)

        plt.figure(**figure_kwargs)
        plt.hist(img.flatten(), bins=bins, **hist_kwargs)
        if self._show:
            plt.show()
        else:
            plt.savefig(
                os.path.join(self._path, f"hist_{get_sample_id(sample_dict).replace('.', '__')}.png")
            )  # save fig
