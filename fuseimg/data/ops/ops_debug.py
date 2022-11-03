import os
from typing import Callable, Optional, Tuple, Any, Dict
from fuse.data.ops.ops_cast import Cast
from fuse.data.utils.sample import get_sample_id
from fuse.utils import NDict
from fuse.utils.file_io.file_io import create_dir
from fuse.data.ops.ops_debug import OpDebugBase
import numpy
import torch
import matplotlib.pyplot as plt
import SimpleITK as sitk
import plotly.graph_objects as go
import numpy as np


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


class OpVis3DPlotly(OpDebugBase):
    """
    Visualize a 3D image as plotly figure exported to html.
    Supports only gray scale.
    To view results, open the <path>_PLOTLY.html file generated in chrome (have not yet tested another browser).
    Example of use:
            (OpVis3DPlotly(num_samples=1), dict(key="data.debug.3d_volume"))
    """

    def __init__(
        self,
        path: str = ".",
        callback: Optional[Callable] = None,
        **kwargs: Dict[str, Any],
    ):
        """
        :param path: location to save .html rendered volume files
        :param callback: function which accepts np.ndarray which can be applied immediately before visualization to the volume
        :param kwargs: see super class arguments
        Example using callback (with some function to binarize the volume):
            (OpVis3DPlotly(num_samples=1, callback=lambda x:np.where(x>0.5, 1, 0)), dict(key="data.debug.3d_volume")),
        """
        super().__init__(**kwargs)
        self._path = path
        self._callback = callback

    def frame_args(self, duration: int) -> dict:
        """
        Required for buttons used in the plotly graph animation.
        Everything is hardcoded/constant except for duration.
        :param duration: length of animation once relevant button is activated
        """

        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    def get_plotly_fig_from_vol(self, vol: np.ndarray) -> go.Figure:
        """
        this function will return a plotly figure (which can later be saved) created from a 3d numpy-like matrix
        :param vol: lxwxh numpy-like matrix
        returns: plotly.graph_objs._figure.Figure
        """
        # uncomment the below line for example from their website (brain MRI)
        # vol = io.imread("<https://s3.amazonaws.com/assets.datacamp.com/blog_assets/attention-mri.tif>")

        nb_frames, w, h = vol.shape
        cmin = vol.min()
        cmax = vol.max()

        fig = go.Figure(
            frames=[
                go.Frame(
                    data=go.Surface(
                        z=(nb_frames - k) * np.ones((w, h)),  # height
                        surfacecolor=np.flipud(vol[(nb_frames - 1) - k]),
                        cmin=cmin,
                        cmax=cmax,
                    ),
                    name=str(k),  # you need to name the frame for the animation to behave properly
                )
                for k in range(nb_frames)
            ]
        )

        # Add data to be displayed before animation starts
        fig.add_trace(
            go.Surface(
                z=nb_frames * np.ones((w, h)),
                surfacecolor=np.flipud(vol[0]),
                colorscale="Gray",
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(thickness=20, ticklen=4),
            )
        )

        sliders = [
            {
                "pad": {"b": 10, "t": 60},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f.name], self.frame_args(0)],
                        "label": str(k),
                        "method": "animate",
                    }
                    for k, f in enumerate(fig.frames)
                ],
            }
        ]

        # Layout
        fig.update_layout(
            title="Slices in volumetric data",
            width=600,
            height=600,
            scene=dict(
                zaxis=dict(range=[-1, nb_frames], autorange=False),
                aspectratio=dict(x=1, y=1, z=1),
            ),
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [None, self.frame_args(50)],
                            "label": "&#9654;",  # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], self.frame_args(0)],
                            "label": "&#9724;",  # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
            ],
            sliders=sliders,
        )
        return fig

    def save_plotly_fig_html(self, fig: go.Figure, sample_dict: NDict) -> None:
        """
        Saves a plotly figure object as html.
        :param fig: the plotly figure object that you want to save
        :param sample_dict: dict containing minibatch sample. Used only to extract sample id when saving to filename
        """
        if self._name is None:
            # filename will be something like ./ProstateX-0008_1_PLOTLY.html
            filename = os.path.join(self._path, get_sample_id(sample_dict).replace(".", "__")) + "_PLOTLY" + ".html"
        else:
            filename = (
                os.path.join(self._path, self._name, get_sample_id(sample_dict).replace(".", "__"))
                + "_PLOTLY"
                + ".html"
            )
        create_dir(os.path.dirname(filename))
        fig.write_html(filename)

    def call_debug(
        self,
        sample_dict: NDict,
        key: str,
    ) -> None:
        """
        :param sample_dict: dict containing minibatch sample
        :param key: sample_dict key to a 3D image. Either tensor or numpy array.
        """

        # ensure vol is np.ndarray
        vol = sample_dict[key]
        if isinstance(vol, torch.Tensor):
            vol = vol.detach().cpu().numpy()
        if isinstance(vol, sitk.Image):  # should not be array! temp sagi
            vol = sitk.GetArrayFromImage(vol)
        assert isinstance(vol, numpy.ndarray)

        if self._callback is not None:
            vol = self._callback(vol)

        # make plotly figure and save
        plotly_fig = self.get_plotly_fig_from_vol(vol)
        self.save_plotly_fig_html(plotly_fig, sample_dict)
