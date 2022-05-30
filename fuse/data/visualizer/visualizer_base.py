from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, List ,Union , Tuple
from fuse.utils.ndict import NDict


class VisualizerBase(ABC):
    
    def __init__(self) -> None:
        """
        initialize the visualizer
        """
        super().__init__()
        
    @abstractmethod
    def _show(self, vis_data: List ):
        """
        actual visualization function, gets a preprocessed collection of items to visualize/compare and shows
        a visualization window that is blocking.
        should be overriden by a specific visualizer
        
        :param vis_data: preprocessed visualization items to display
        """
        raise "should implement abstract method"
    
    @abstractmethod
    def _save(self, vis_data: List , output_path : str , file_name : str):
        """
        actual visualization function, gets a preprocessed collection of items to visualize/compare and saves the visualization to file
        should be overriden by a specific visualizer
        
        :param vis_data: preprocessed visualization items to display
        :param output_path : output folder location
        :param file_name : file name
        """
        raise "should implement abstract method"
    
    @abstractmethod
    def _preprocess(self, vis_data: Union[List, NDict]) -> Tuple[List[Dict] , str]:
        """
        get the collected data from the sample, that the visProbe has collected and generated data for actual visualization
        that the _show method can process
        
        :param vis_data: the collected data
        """
        raise "should implement abstract method"
    
    def show(self, vis_data: List ):
        data  = self._preprocess(vis_data )
        self._show(data )
        
    def save(self,  vis_data: List , op_id: str , output_path : str ):
        data = self._preprocess(vis_data )
        self._save(data , output_path , op_id)