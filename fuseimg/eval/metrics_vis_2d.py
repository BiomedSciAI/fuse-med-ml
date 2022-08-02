from functools import partial
from typing import Any, Callable, Dict, Hashable, Optional, Sequence
from fuse.eval.metrics.metrics_common import MetricWithCollectorBase
from fuseimg.data.ops.ops_debug import OpVis2DImage


class MetricVis2DImage(MetricWithCollectorBase):
    """
    Visualize a 2D image (either display it or save it as an image).
    Supports both rgb and gray scale and supports few image formats "channels_first"/"no_channels"/"channels_last"
    """

    def __init__(
        self,
        sample_ids: Optional[Sequence[Hashable]] = None,
        num_samples: Optional[int] = None,
        show: bool = True,
        path: str = ".",
        image_format: str = "channels_first",
        image_process_func: Optional[Callable] = None,
        **kwargs: dict,
    ) -> None:
        """
        :param sample_ids: apply for the specified sample ids. To apply for all set to None.
        :param num_samples: apply for the first num_samples (per process). if None, will apply for all.
        :param show: if set to true will display it, otherwise will save it in `path`
        :param path: location to save the images. Used when `show` set to False
        :param image_format: the format of the image: "channels_first"/"no_channels"/"channels_last"
        :param image_processing_func: callable that get the image (numpy array) and can process it. Typically use to move the range to values matplotlib can display.
        :param kwargs: See OpVis2DImage.call_debug() arguments
        """
        self._op_vis = OpVis2DImage(
            sample_ids=sample_ids,
            num_samples=num_samples,
            show=show,
            path=path,
            image_format=image_format,
            image_process_func=image_process_func,
        )

        op_vis_func = partial(self._op_vis, **kwargs)
        super().__init__(pre_collect_process_func=op_vis_func)
        self._epoch = -1

    def reset(self) -> None:
        self._epoch += 1
        super().reset()
        self._op_vis.reset(f"epoch_{self._epoch}")

    def eval(self, results: Dict[str, Any] = None, ids: Optional[Sequence[Hashable]] = None) -> None:
        # Do nothing - the logic run pre sample in pre_collect_process func
        pass
