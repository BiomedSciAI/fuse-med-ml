from typing import List , Callable , Any
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import matplotlib.patches as patches
from fuseimg.utils.typing.typed_element import TypedElement
from itertools import cycle
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

def show_multiple_images(plot_label : Callable , imgs : List, base_resolution :int = 5, **args ):
    '''
    Show multiple images with shared zoom/translation controls
    everything passed as kwargs (for example - cmap='gray') will be passed to the individual imshow calls

    special possible values are:
        * cmap[image_index] = value
            for example - cmap0='gray' will only change the cmap of the first image
        * unify_size = 'dont_care'
            will make sure that all images are resized to the same size

    example usage:

    imshowmultiple([img1, img2])
    imshowmultiple(img1,img2,img3, cmap='gray', vmin=0.0, vmax=1.0)
    imshowmultiple(img1,img2,img3, cmap0='gray')   #will use grayscale color map only on the first image and default cmap on the rest
    imshowmultiple(img1,img2,img3, unify_size='blah')   #will resize all images to match

    @param plot_label : function to plot the ground truth segmentation
    @param imgs: list of images in TypedElement format
    @param base_resolution : base pixel resolution we want to maintain per image
    @param args: additional parameters to the plot function of matplotlib
    @return:
    '''
    assert len(imgs)>0
    grid_size = int(np.sqrt(len(imgs))) + 1
    fig = plt.figure(figsize=(base_resolution*grid_size,base_resolution*grid_size))
    axis = []
    
    do_not_pass = ['unify_size','color']
    do_not_pass += [ 'cmap'+str(i) for i in range(20)]
    pass_kwargs = { k:d for k,d in args.items() if k not in do_not_pass }
    for i,m in enumerate(imgs):

        im = m.image
        if 0==i:
            axis.append(fig.add_subplot(grid_size,grid_size,i+1))
        else:
            axis.append(fig.add_subplot(grid_size, grid_size, i + 1, sharex=axis[0],sharey=axis[0]))
            
        img_cmap = 'cmap'+str(i+1)
        if img_cmap in args:
            pass_kwargs['cmap'] = args[img_cmap]      
            axis[i].imshow(im, interpolation='none', **pass_kwargs)
        else:
            axis[i].imshow(im, interpolation='none', **pass_kwargs)
           
        plot_label(axis[i], m , args['color'])
                
        if m.metadata:
            axis[i].set_title(str(i)+":"+m.metadata)
        else:
            axis[i].set_title(str(i))

    return fig

def plot_color_mask(mask , ax ) :
    color_mask = np.random.random((1, 3)).tolist()[0]
    masked = np.ma.masked_where(mask == 0, mask)
    img = np.ones( (masked.shape[0], masked.shape[1], 3) )
    for i in range(3):
        img[:,:,i] = color_mask[i]
    ax.imshow(np.dstack( (img, masked*0.5) )) 
    
    
def plot_seg(ax : Any, sample : TypedElement, color ):
    if sample.seg is not None : 
        mask = sample.seg
        masked = np.ma.masked_where(mask == 0, mask)
        ax.imshow(masked) 
    polygons = []
    colors = []
    cycol = cycle(color)
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
             
    

show_multiple_images_seg = partial(show_multiple_images, plot_label=plot_seg)
