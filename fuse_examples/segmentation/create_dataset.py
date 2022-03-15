# from fastai.vision import *
import pydicom
from pathlib import Path
import pandas as pd
from tqdm import tqdm as progress_bar
import PIL
import numpy as np
import matplotlib.pylab as plt
# from mask_functions import *
# Here out RLE encoding is little bit different

import sys
sys.path.append('JustImage/')
sys.path.append('JustImage/CV')
from JustImage.CV.Binarization import bin_mosaic1, bin_mosaic3

main_out_path = 'data' 
dataset_path = 'siim'

"""
download dataset from - 
https://www.kaggle.com/seesee/siim-train-test

The path to the extracted data should be updated in the <dataset_path> variable.
The output images will be stored at <main_out_path>.
"""

def create_mosaic(im, DEBUG=False):
    # im = im.astype(np.uint8)
    im = np.asarray(im)
    org_min = np.min(im)
    org_max = np.max(im)

    _, im_after = bin_mosaic1(im, tiles=(8, 8), locality_size=8)
    
    s = (im_after - np.min(im_after)) / (np.max(im_after) - np.min(im_after))
    out_im = s * (org_max - org_min) + org_min
    out_im = out_im.astype(np.uint8)

    _, im_after = bin_mosaic1(im, tiles=(16, 16), locality_size=8)
    
    s = (im_after - np.min(im_after)) / (np.max(im_after) - np.min(im_after))
    out_im2 = s * (org_max - org_min) + org_min
    out_im2 = out_im2.astype(np.uint8)

    rgb = np.stack([im, out_im, out_im2], axis=-1)

    if DEBUG:
        fig, ax = plt.subplots(2, 3, figsize=(15, 8))
        ax[0, 0].imshow(im, cmap=plt.cm.bone)
        ax[0, 1].imshow(out_im, cmap=plt.cm.bone)
        ax[0, 2].imshow(out_im2, cmap=plt.cm.bone)

        ax[1, 0].hist(im.ravel()*255, np.arange(0,255));
        ax[1, 1].hist(out_im.ravel(), np.arange(np.min(out_im), np.max(out_im)));
        ax[1, 2].hist(out_im2.ravel(), np.arange(np.min(out_im2), np.max(out_im2)));

        plt.figure()
        plt.imshow(rgb)
        plt.show()

    return PIL.Image.fromarray(rgb)

def rle2mask(rles, width, height):
    """
    
    rle encoding if images
    input: rles(list of rle), width and height of image
    returns: mask of shape (width,height)
    """
    
    mask= np.zeros(width* height)
    for rle in rles:
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]

        current_position = 0
        for index, start in enumerate(starts):
            current_position += start
            mask[current_position:current_position+lengths[index]] = 255
            current_position += lengths[index]

    return mask.reshape(width, height).T

def filter_files(files, include=[], exclude=[]):
    for incl in include:
        files = [f for f in files if incl in f.name]
    for excl in exclude:
        files = [f for f in files if excl not in f.name]
    return sorted(files)

def ls(x, recursive=False, include=[], exclude=[]):
    if not recursive:
        out = list(x.iterdir())
    else:
        out = [o for o in x.glob('**/*')]
    out = filter_files(out, include=include, exclude=exclude)
    return out

Path.ls = ls

class InOutPath():
    def __init__(self, input_path:Path, output_path:Path):
        if isinstance(input_path, str): input_path = Path(input_path)
        if isinstance(output_path, str): output_path = Path(output_path)
        self.inp = input_path
        self.out = output_path
        self.mkoutdir()

    def mkoutdir(self):
        self.out.mkdir(exist_ok=True, parents=True)

    def __repr__(self):
        return '\n'.join([f'{i}: {o}' for i, o in self.__dict__.items()]) + '\n'

def dcm2png(SZ, dataset):
    path = InOutPath(Path(dataset_path + f'/dicom-images-{dataset}'), Path(main_out_path + f'/data{SZ}/{dataset}'))
    files = path.inp.ls(recursive=True, include=['.dcm'])
    for f in progress_bar(files):
        dcm = pydicom.read_file(str(f)).pixel_array
        # PIL.Image.fromarray(dcm).resize((SZ,SZ)).save(path.out/f'{f.stem}.png')

        # make a rgb like image with mosaic images:
        im = PIL.Image.fromarray(dcm).resize((SZ,SZ))
        im = create_mosaic(im)
        im.save(path.out/f'{f.stem}.png')

def masks2png(SZ):
    path = InOutPath(Path('data'), Path(main_out_path + f'/data{SZ}/masks'))
    for i in progress_bar(list(set(rle_df.ImageId.values))):
        I = rle_df.ImageId == i
        name = rle_df.loc[I, 'ImageId']
        enc = rle_df.loc[I, ' EncodedPixels']
        if sum(I) == 1:
            enc = enc.values[0]
            name = name.values[0]
            if enc == '-1': # ' -1':
                m = np.zeros((1024, 1024)).astype(np.uint8)
            else:
                m = rle2mask([enc], 1024, 1024).astype(np.uint8)
            PIL.Image.fromarray(m).resize((SZ,SZ)).save(f'{path.out}/{name}.png')
        else:
            # m = np.array([rle2mask(e, 1024, 1024).astype(np.uint8) for e in enc.values])
            m = rle2mask(enc.values, 1024, 1024).astype(np.uint8)
            # m = m.sum(0).astype(np.uint8).T
            PIL.Image.fromarray(m).resize((SZ,SZ)).save(f'{path.out}/{name.values[0]}.png')

rle_df = pd.read_csv(dataset_path + '/train-rle.csv')

size_list = [128]  # [64, 128, 256, 512, 1024]
for SZ in progress_bar(size_list):
    print(f'Converting data for train{SZ}')
    dcm2png(SZ, 'train')
    print(f'Converting data for test{SZ}')
    dcm2png(SZ, 'test')
    print(f'Generating masks for size {SZ}')
    masks2png(SZ)

for SZ in progress_bar(size_list):
    # Missing masks set to 0
    print('Generating missing masks as zeros')
    train_images = [o.name for o in Path(main_out_path + f'/data{SZ}/train').ls(include=['.png'])]
    train_masks = [o.name for o in Path(main_out_path + f'/data{SZ}/masks').ls(include=['.png'])]
    missing_masks = set(train_images) - set(train_masks)
    path = InOutPath(Path('data'), Path(main_out_path + f'/data{SZ}/masks'))
    for name in progress_bar(missing_masks):
        m = np.zeros((1024, 1024)).astype(np.uint8).T
        PIL.Image.fromarray(m).resize((SZ,SZ)).save(main_out_path + f'/data{SZ}/masks/{name}')
