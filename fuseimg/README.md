# FuseMedML - extension for imaging

## Data package
Extends FuseMedML [data package](../fuse/data/README.md) for imaging.
Here you can find useful ops for imaging and public datasets implementation

### Imaging operators

### [color](data/ops/color.py)
* OpClip - clip values pixels/voxels values, support both torch tensor and numpy array
* OpNormalizeAgainstSelf - normalizes a tensor into [0.0, 1.0] using its img.min() and img.max() (NOT against a dataset)
* OpToIntImageSpace - normalizes a tensor into [0, 255] int gray-scale using img.min() and img.max() (NOT against a dataset)
* OpToRange - linearly project from a range to a different range

#### [image loader](data/ops/image_loader.py)
* OpLoadImage - loads variety of medical imaging formats from disk


#### [shape](data/ops/shape_ops.py)
* OpHWCToCHW - transform HWC (height, width, channel) to CHW (channel, height, width)
* OpCHWToHWC - transform CHW (channel, height, width) to HWC (height, width, channel)
* OpSelectSlice - select one slice from the input tensor, 

#### [augmentation](data/ops/aug/)

[**color**](data/ops/aug/color.py)
* OpAugColor - color augmentation for gray scale images of any dimensions, including addition, multiplication, gamma and contrast adjusting 
* OpAugGaussian - add gaussian noise to numpy array or torch tensor of any dimensions
  
[**geometry**](data/ops/aug/geometry.py)
* OpAugAffine2D -  2D affine transformation for torch tensors
* OpAugCropAndResize2D - alternative to rescaling in OpAugAffine2D: center crop and resize back to the original dimensions. if scale is bigger than 1.0. the image first padded.
* OpAugSqueeze3Dto2D - squeeze selected axis of volume image into channel dimension, in order to fit the 2D augmentation functions
* OpAugUnsqueeze3DFrom2D - unsqueeze back to 3D, after you apply the required 2D operations

### Imaging datasets
* [kits21](datasets/kits21.py) - 2021 Kidney and Kidney Tumor Segmentation Challenge Dataset. See https://github.com/neheller/kits21
* [stoic21](datasets/stoic21.py) - Dataset created for COVID-19 AI challenge - https://stoic2021.grand-challenge.org/. Aims to predict the severe outcome of COVID-19, based on the largest dataset of Computed Tomography (CT) images of COVID-19
* [mnist](datasets/mnist.py) - mnist dataset implementation
