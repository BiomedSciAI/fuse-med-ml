from typing import List, Optional, Union
import enum

from fuse.utils.ndict import NDict
from fuse.data.visualizer.visualizer_base import VisualizerBase
from .op_base import OpReversibleBase
from fuse.data.key_types import TypeDetectorBase


class VisFlag(enum.IntFlag):
    COLLECT = 1  # save current state for future comparison
    SHOW_CURRENT = 2  # show current state
    SHOW_COLLECTED = 4  # show comparison of all previuosly collected states
    CLEAR = 8  # clear all collected states until this point in the pipeline
    ONLINE = 16  # show operations will prompt the user with the releveant plot
    OFFLINE = 32  # show operations will write to disk (using the caching mechanism) the relevant info (state or states for comparison)
    FORWARD = 64  # visualization operation will be activated on forward pipeline execution flow
    REVERSE = 128  # visualization operation will be activated on reverse pipeline execution flow
    SHOW_ALL_COLLECTED = 256  # show comparison of all previuosly collected states


class VisProbe(OpReversibleBase):
    """
    Handle visualization, saves, shows and compares the sample with respect to the current state inside a pipeline
    In most cases VisProbe can be used regardless of the domain, and the domain specific code will be implemented
    as a Visualizer inheriting from VisualizerBase. In some cases there might be need to also inherit from VisProbe.

    Important notes:
    - running in a cached environment is dangerous and is prohibited
    - this Operation is not thread safe ans so multithreading is also discouraged

    "
    """

    def __init__(
        self,
        flags: VisFlag,
        keys: Union[List, dict],
        type_detector: TypeDetectorBase,
        id_filter: Union[None, List] = None,
        visualizer: VisualizerBase = None,
        cache_path: str = "~/",
    ):
        """
        :param flags: operation flags (or possible concatentation of flags using IntFlag), details:
            COLLECT - save current state for future comparison
            SHOW_CURRENT - show current state
            SHOW_COllected - show comparison of all previuosly collected states
            CLEAR - clear all collected states until this point in the pipeline
            ONLINE - show operations will prompt the user with the releveant plot
            OFFLINE - show operations will write to disk (using the caching mechanism) the relevant info (state or states for comparison)
            FORWARD - visualization operation will be activated on forward pipeline execution flow
            REVERSE - visualization operation will be activated on reverse pipeline execution flow
        :param keys: for which sample keys to handle visualization, also can be grouped in a dictionary
        :param id_filter: for which sample id's to be activated, if None, active for all samples
        :param visualizer: the actual visualization handler, depands on domain and use case, should implement Visualizer Base
        :param cache_path: root dir to save the visualization outputs in offline mode

        few issues to be aware of, detailed in github issues regarding static cached pipeline and multiprocessing
        note - if both forward and reverse are on, then by default, on forward we do collect and on reverse we do show_collected to
        compare reverse operations
        for each domain we inherit for VisProbe like ImagingVisProbe,..."""
        super().__init__()
        self._id_filter = id_filter
        self._keys = keys
        self._flags = flags
        self._cacher = None
        self._collected_prefix = "data.$vis"
        self._cache_path = cache_path
        self._visualizer = visualizer
        self._type_detector = type_detector

    def _extract_collected(self, sample_dict: NDict):
        res = []
        if self._collected_prefix not in sample_dict:
            return res
        else:
            for vdata in sample_dict[self._collected_prefix]:
                res.append(vdata)
        return res

    def _extract_data(self, sample_dict: NDict, keys, op_id):
        if type(keys) is list:
            # infer keys groups
            keys.sort()
            first_type = self._type_detector.get_type(sample_dict, keys[0])
            num_of_groups = len(
                [
                    self._type_detector.get_type(sample_dict, k)
                    for k in keys
                    if self._type_detector.get_type(sample_dict, k) == first_type
                ]
            )
            keys_per_group = len(keys) // num_of_groups
            keys = {f"group{i}": keys[i : i + keys_per_group] for i in range(0, len(keys), keys_per_group)}

        res = NDict()
        for group_id, group_keys in keys.items():
            for key in group_keys:
                prekey = f'groups.{group_id}.{key.replace(".", "_")}'
                res[f"{prekey}.value"] = sample_dict[key]
                res[f"{prekey}.type"] = self._type_detector.get_type(sample_dict, key)
        res["$step_id"] = op_id
        return res

    def _save(self, vis_data: Union[List, dict]):
        # use caching to save all relevant vis_data
        print("saving vis_data", vis_data)

    def _handle_flags(self, flow, sample_dict: NDict, op_id: Optional[str]):
        """
        See super class
        """
        # sample was filtered out by its id
        if self._id_filter and self.get_idx(sample_dict) not in self._id_filter:
            return None
        if flow not in self._flags:
            return None

        # grouped key dictionary with the following structure:
        # vis_data = {"cc_group":
        #                {
        #                    "key1": {
        #                      "value": ndarray,
        #                      "type": DataType.Image,
        #                      "op_id": "test1"}
        #                    "key2": {
        #                      "value": ndarray,
        #                      "type": DataType.BBox,
        #                      "op_id": "test1"}
        #                 },
        #            "mlo_goup":
        #                {
        #                    "key3": {
        #                      "value": ndarray,
        #                      "type": DataType.Image,
        #                      "op_id": "test1"}
        #                    "key4": {
        #                      "value": ndarray,
        #                      "type": DataType.BBox,
        #                      "op_id": "test1"}
        #                 },
        #            }
        vis_data = self._extract_data(sample_dict, self._keys, op_id)
        both_fr = (VisFlag.REVERSE | VisFlag.FORWARD) in self._flags
        dir_forward = flow == VisFlag.FORWARD
        dir_reverse = flow == VisFlag.REVERSE
        # any_show_collected = VisFlag.SHOW_ALL_COLLECTED | VisFlag.SHOW_COLLECTED

        if VisFlag.COLLECT in self._flags or (dir_forward and both_fr):
            if self._collected_prefix not in sample_dict:
                sample_dict[self._collected_prefix] = []
            sample_dict[self._collected_prefix].append(vis_data)

        if VisFlag.SHOW_CURRENT in self._flags:
            if VisFlag.ONLINE in self._flags:
                self._visualizer.show(vis_data)
            if VisFlag.OFFLINE in self._flags:
                self._save(vis_data)

        if (VisFlag.SHOW_ALL_COLLECTED in self._flags or VisFlag.SHOW_COLLECTED in self._flags) and (
            (both_fr and dir_reverse) or not both_fr
        ):
            vis_data = self._extract_collected(sample_dict) + [vis_data]
            if both_fr:
                if VisFlag.SHOW_COLLECTED in self._flags:
                    vis_data = vis_data[-2:]
            if VisFlag.ONLINE in self._flags:
                self._visualizer.show(vis_data)
            if VisFlag.OFFLINE in self._flags:
                self.save(vis_data)

        if VisFlag.CLEAR in self._flags:
            sample_dict[self._collected_prefix] = []

        if VisFlag.SHOW_COLLECTED in self._flags and both_fr and dir_reverse:
            sample_dict[self._collected_prefix].pop()

        return sample_dict

    def __call__(self, sample_dict: NDict, op_id: Optional[str], **kwargs) -> Union[None, dict, List[dict]]:
        res = self._handle_flags(VisFlag.FORWARD, sample_dict, op_id)
        return res

    def reverse(self, sample_dict: NDict, op_id: Optional[str], key_to_reverse: str, key_to_follow: str) -> dict:
        """
        See super class
        """
        res = self._handle_flags(VisFlag.REVERSE, sample_dict, op_id)
        if res is None:
            res = sample_dict
        return res
