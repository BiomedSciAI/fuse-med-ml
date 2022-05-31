from typing import List , Callable , Any
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from fuseimg.utils.typing.typed_element import TypedElement
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

def show_multiple_images(plot_seg : Callable , imgs : List, base_resolution :int = 5, **kwargs ) -> Figure:
    '''
    Show multiple images with shared zoom/translation controls
    everything passed as kwargs (for example - cmap='gray') will be passed to the individual imshow calls
    :param plot_seg : function to plot the ground truth segmentation
    :param imgs: list of images in TypedElement format
    :param base_resolution : base pixel resolution we want to maintain per image
    :param kwargs: additional parameters to the plot function of matplotlib
    :return: matplotlib figure object with the images grid and segmentation
    '''
    assert len(imgs)>0
    grid_size = int(np.sqrt(len(imgs))) + 1
    fig = plt.figure(figsize=(base_resolution*grid_size,base_resolution*grid_size))
    axis = []
    
    for i,m in enumerate(imgs):
        im = m.image
        if i == 0 :
            axis.append(fig.add_subplot(grid_size,grid_size,i+1))
        else:
            axis.append(fig.add_subplot(grid_size, grid_size, i + 1, sharex=axis[0],sharey=axis[0]))
        axis[i].imshow(im, **kwargs) 
        plot_seg(axis[i], m )      
        if m.metadata:
            axis[i].set_title(str(i)+":"+m.metadata)
        else:
            axis[i].set_title(str(i))

    return fig

 
def plot_seg_coco_style(ax : Any, sample : TypedElement ):
    if sample.seg is not None : 
        mask = sample.seg
        masked = np.ma.masked_where(mask == 0, mask)
        ax.imshow(masked) 
    polygons = []
    colors = []
    if sample.contours is not None :
        for seg in sample.contours :
            seg = np.array(seg[0])
            poly = seg.reshape(int(len(seg)/2), 2)
            polygons.append(Polygon(poly))
            c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
            colors.append(c)
    if sample.bboxes is not None :
        for bbox in sample.bboxes :
            [bbox_x, bbox_y, bbox_w, bbox_h] = bbox
            poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
            np_poly = np.array(poly).reshape((4,2))
            polygons.append(Polygon(np_poly))
            c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
            colors.append(c)
    p = PatchCollection(polygons, facecolor=colors, linewidths=0, alpha=0.4)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', edgecolors=colors, linewidths=2)
    ax.add_collection(p)
             
    

show_multiple_images_seg = partial(show_multiple_images, plot_seg=plot_seg_coco_style)