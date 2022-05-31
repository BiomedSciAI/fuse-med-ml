from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, List ,Union , Tuple
from fuse.utils.ndict import NDict


class VisualizerBase(ABC):
    
    def __init__(self, pre_process_func: Callable = None) -> None:
        """
        initialize the visualizer
        """
        super().__init__()
        self._pre_process = pre_process_func
        
    @abstractmethod
    def _show(self, vis_data: List[Dict[str, str]] ):
        """
        actual visualization function, gets a preprocessed collection of items to visualize/compare and shows
        a visualization window that is blocking.
        should be overriden by a specific visualizer
        
        :param vis_data: preprocessed visualization items to display
        """
        raise NotImplementedError
    
    @abstractmethod
    def _save(self, vis_data: List , output_path : str , file_name : str):
        """
        actual visualization function, gets a preprocessed collection of items to visualize/compare and saves the visualization to file
        should be overriden by a specific visualizer
        
        :param vis_data: preprocessed visualization items to display
        :param output_path : output folder location
        :param file_name : file name
        """
        raise NotImplementedError
    
    
    def show(self, vis_data: List ):
        if self._pre_process != None :
            vis_data  = self._preprocess(vis_data )
        self._show(vis_data )
        
    def save(self,  vis_data: List , op_id: str , output_path : str ):
        if self._pre_process != None :
            vis_data  = self._preprocess(vis_data )
        self._save(vis_data , output_path , op_id)