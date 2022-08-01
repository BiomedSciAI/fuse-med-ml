from abc import ABC, abstractmethod
from typing import Dict, Any, List


class VisualizerBase(ABC):
    def __init__(self) -> None:
        super().__init__()

    def _preprocess(self, vis_data: Dict[str, Any]):
        """
        get the collected data from the sample, that the visProbe has collected and generated data for actual visualization
        that the _show method can process

        :param vis_data: the collected data
        """
        return vis_data

    @abstractmethod
    def _show(self, vis_data: List):
        """
        actual visualization function, gets a preprocessed collection of items to visualize/compare and shows
        a visualization window that is blocking.
        should be overriden by a specific visualizer

        :param vis_data: preprocessed visualization items to display
        """
        raise "should implement abstract method"

    def show(self, vis_data):
        data = self._preprocess(vis_data)
        self._show(data)


class PrintVisual(VisualizerBase):
    """
    basic visualizer example that just prints the data string representation to the console
    """

    def __init__(self) -> None:
        super().__init__()

    def _show(self, vis_data):
        if type(vis_data) is dict:
            print("showing single item")
            print(vis_data)
        else:
            print(f"comparing {len(vis_data)} items:")
            for item in vis_data:
                print(item)

    def show(self, vis_data):
        data = self._preprocess(vis_data)
        self._show(data)
