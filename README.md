[![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg)](https://opensource.org/)
[![PyPI version](https://badge.fury.io/py/fuse-med-ml.svg)](https://badge.fury.io/py/fuse-med-ml)
[![Python version](https://img.shields.io/pypi/pyversions/fuse-med-ml)](https://pypi.org/project/fuse-med-ml/)
[![Slack channel](https://img.shields.io/badge/support-slack-slack.svg?logo=slack)](https://join.slack.com/t/fusemedml/shared_invite/zt-xr1jaj29-h7IMsSc0Lq4qpVNxW97Phw)
[![Downloads](https://pepy.tech/badge/fuse-med-ml)](https://pepy.tech/project/fuse-med-ml)


<img src="fuse/doc/FuseMedML-logo.png" alt="drawing" width="30%"/>

# Effective Code Reuse across ML projects!

A python framework accelerating ML based discovery in the medical field by encouraging code reuse. Batteries included :)

## Jump to:

* install instructions [section](#installation)
* complete code [examples](#examples)
* [community support](#community-support---join-the-discussion)
* Contributing to FuseMedML [guide](./CONTRIBUTING.md)
* [citation info](#citation)



# Motivation - *"*Oh, the pain!*"*
Analyzing **many** ML research projects we discovered that
* Projects bring up is taking **far too long**, even when very similar projects were already done in the past by the same lab!
* Porting individual components across projects was *painful* - resulting in **"reinventing the wheel" time after time**

# How the magic happens

## 1. A simple yet super effective design concept
### Data is kept in a nested (hierarchical) dictionary
This is a key aspect in FuseMedML (shortly named as "fuse"). It's a key driver of flexiblity, and allows to easily deal with multi modality information.
```python
from fuse.utils import NDict

sample_ndict = NDict()
sample_ndict['input.mri'] = # ...
sample_ndict['input.ct_view_a'] = # ...
sample_ndict['input.ct_view_b'] = # ...
sample_ndict['groundtruth.disease_level_label'] = # ...
```

This data can be a single sample, it can be for a minibatch, for an entire epoch, or anything that is desired.
The "nested key" ("a.b.c.d.etc') is called "path key", as it can be seen as a path inside the nested dictionary.

**Components are written in a way that allows to define input and output keys, to be read and written from the nested dict**
See a short introduction video (3 minutes) to how FuseMedML components work:

https://user-images.githubusercontent.com/7043815/177197158-d3ea0736-629e-4dcb-bd5e-666993fbcfa2.mp4


### Examples - using FuseMedML-style components


A multi head model FuseMedML style component, allows easy reuse across projects:

```python
ModelMultiHead(
    conv_inputs=(('data.input.img', 1),),                                       # input to the backbone model
    backbone=BackboneResnet3D(in_channels=1),                                   # PyTorch nn Module
    heads=[                                                                     # list of heads - gives the option to support multi task / multi head approach
               Head3D(head_name='classification',
                                mode="classification",
                                conv_inputs=[("model.backbone_features", 512)]  # Input to the classification head
                                ,),
          ]
)
```

Our default loss implementation - creates an easy wrap around a callable function, while being FuseMedML style
```python
LossDefault(
    pred='model.logits.classification',          # input - model prediction scores
    target='data.label',                         # input - ground truth labels
    callable=torch.nn.functional.cross_entropy   # callable - function that will get the prediction scores and labels extracted from batch_dict and compute the loss
)
```

An example metric that can be used
```python
MetricAUCROC(
    pred='model.output', # input - model prediction scores
    target='data.label'  # input - ground truth labels
)
```

Note that several components return answers directly and not write it into the nested dictionary. This is perfectly fine, and to allow maximum flexibility we do not require any usage of output path keys.

### Creating a custom FuseMedML component

Creating custom FuseMedML components is easy - in the following example we add a new data processing operator:


A data pipeline operator
```python
class OpPad(OpBase):
    def __call__(self, sample_dict: NDict,
        key_in: str,
        padding: List[int], fill: int = 0, mode: str = 'constant',
        key_out:Optional[str]=None,
        ):

        #we extract the element in the defined key location (for example 'input.xray_img')
        img = sample_dict[key_in]
        assert isinstance(img, np.ndarray), f'Expected np.ndarray but got {type(img)}'
        processed_img = np.pad(img, pad_width=padding, mode=mode, constant_values=fill)

        #store the result in the requested output key (or in key_in if no key_out is provided)
        key_out = key_in if key_out is None
        sample_dict[key_out] = processed_img

        #returned the modified nested dict
        return sample_dict
```

Since the key location isn't hardcoded, this module can be easily reused across different research projects with very different data sample structures. More code reuse - Hooray!

FuseMedML-style components in general are any classes or functions that define which key paths will be written and which will be read.
Arguments can be freely named, and you don't even have to write anything to the nested dict.
Some FuseMedML components return a value directly - for example, loss functions.

## 2. "Batteries included" key components, built using the same design concept

### **[fuse.data](./fuse/data)** - A **declarative** super flexible data processing pipeline
* Easy dealing with complex multi modality scenario
* Advanced caching, including periodic audits to automatically detect stale caches
* Default ready-to-use Dataset and Sampler classes
* See detailed introduction [here](./fuse/data/README.md)

### **[fuse.eval](./fuse/eval)** - a standalone library for **evaluating ML models** (not necessarily trained with FuseMedML)
The package includes collection of off-the-shelf metrics and utilities such as **statistical significance tests, calibration, thresholding, model comparison** and more.
See detailed introduction [here](./fuse/eval/README.md)

### **[fuse.dl](./fuse/dl)** - reusable dl (deep learning) model architecture components, loss functions, etc.


## Supported DL libraries
Some components depend on pytorch. For example, ```fuse.data``` is oriented towards pytorch DataSet, DataLoader, DataSampler etc.
```fuse.dl``` makes heavy usage of pytorch models.
Some components do not depend on any specific DL library - for example ```fuse.eval```.

Broadly speaking, the supported DL libraries are:
* "Pure" [pytorch](https://pytorch.org/)
* [pytorch-lightning](https://www.pytorchlightning.ai/)

Before you ask - **pytorch-lightning and FuseMedML play along very nicely and have in practice orthogonal and additive benefits :)**
See [Simple FuseMedML + PytorchLightning Example](./examples/fuse_examples/imaging/classification/mnist/run_mnist.py) for simple supervised learning cases, and [this example ](./examples/fuse_examples/imaging/classification/mnist/run_mnist_custom_pl_imp.py) for completely custom usage of pytorch-lightning and FuseMedML - useful for advanced scenarios such as Reinforcement Learning and generative models.

## Domain Extensions
fuse-med-ml, the core library, is completely domain agnostic!
Domain extensions are optionally installable packages that deal with specific (sub) domains. For example:

* **[fuseimg](./fuseimg)** which was battletested in many medical imaging related projects (different organs, imaging modalities, tasks, etc.)
* **fusedrug (to be released soon)** which focuses on molecular biology and chemistry - prediction, generation and more

Domain extensions contain concrete implementation of components and components parts within the relevant domain, for example:
* [Data pipeline operations](./fuse/data) - for example, a 3d affine transformation of a 3d image
* [Evaluation metrics](./fuse/eval) - for example, a custom metric evaluating docking of a potential drug with a protein target
* [Loss functions](./fuse/dl) - for example, a custom segmentation evaluation loss

The recommended directory structure mimics fuse-med-ml core structure
```
your_package
    data #everything related to datasets, samplers, data processing pipeline Ops, etc.
    dl #everything related to deep learning architectures, optimizers, loss functions etc.
    eval #evaluation metrics
    utils #any utilities
```

You are highly encouraged to create additional domain extensions and/or contribute to the existing ones!
There's no need to wait for any approval, you can create domain extensions on your own repos right away

Note - in general, we find it helpful to follow the same directory structure shown above even in small and specific research projects that use FuseMedML for consistency and easy landing for newcomers into your project :)

# Installation

FuseMedML is tested on Python >= 3.7 and PyTorch >= 1.5

## We recommend using a Conda environment

Create a conda environment using the following command (you can replace FUSEMEDML with your preferred enviornment name)
```bash
conda create -n FUSEMEDML python=3.7
conda activate FUSEMEDML
```

and then do Option 1 or Option 2 below inside the activated conda env



## Option 1: Install from source (recommended)
The best way to install `FuseMedML` is to clone the repository and install it in an [editable mode](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs) using `pip`:
```bash
$ pip install -e .[all]
```
This mode installs all the currently publicly available domain extensions - fuseimg as of now, fusedrug will be added soon.

In this mode you can also install and run our end to end examples using:
```bash
$ pip install -e examples
```

## Option 2: Install from PyPI (does not include examples)
```bash
$ pip install fuse-med-ml[all]
```

# Examples

* Easy access "Hello World" [colab notebook](https://colab.research.google.com/github/BiomedSciAI/fuse-med-ml/blob/master/examples/fuse_examples/imaging/hello_world/hello_world.ipynb)
* classification
    * [**MNIST**](./examples/fuse_examples/imaging/classification/mnist/)  - a simple example, including training, inference and evaluation over [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
    * [**STOIC**](./examples/fuse_examples/imaging/classification/stoic21/) - severe COVID-19 classifier baseline given a Computed-Tomography (CT), age group and gender. [Challenge description](https://stoic2021.grand-challenge.org/)


    * [**KNIGHT Challenge**](./examples/fuse_examples/imaging/classification/knight) - preoperative prediction of risk class for patients with renal masses identified in clinical Computed Tomography (CT) imaging of the kidneys. Including data pre-processing, baseline implementation and evaluation pipeline for the challenge.
    * [**Multimodality tutorial**](https://colab.research.google.com/github/BiomedSciAI/fuse-med-ml/blob/master/examples/fuse_examples/multimodality/image_clinical/multimodality_image_clinical.ipynb) - demonstration of two popular simple methods integrating imaging and clinical data (tabular) using FuseMedML
    * [**Skin Lesion**](./examples/fuse_examples/imaging/classification/isic/) - skin lesion classification , including training, inference and evaluation over the public dataset introduced in [ISIC challenge](https://challenge.isic-archive.com/landing/2019)
    * [**Breast Cancer Lesion Classification**](./examples/fuse_examples/imaging/classification/cmmd) - lesions classification of tumor ( benign, malignant) in breast mammography over the public dataset introduced in [The Chinese Mammography Database (CMMD)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230508)

## Walkthrough template
* [**Walkthrough Template**](./fuse/dl/templates/walkthrough_template.py) - includes several TODO notes, marking the minimal scope of code required to get your pipeline up and running. The template also includes useful explanations and tips.

# Community support - join the discussion!

* Slack workspace at fusemedml.slack.com for informal communication - click [here](https://join.slack.com/t/fusemedml/shared_invite/zt-xr1jaj29-h7IMsSc0Lq4qpVNxW97Phw) to join
* [Github Discussions](https://github.com/BiomedSciAI/fuse-med-ml/discussions)

# Citation
If you use FuseMedML in scientific context, please consider citing us:
```bibtex
@misc{https://doi.org/10.5281/zenodo.5146491,
  doi = {10.5281/ZENODO.5146491},
  url = {https://zenodo.org/record/5146491},
  author = {IBM Research,  Haifa},
  title = {FuseMedML: https://github.com/BiomedSciAI/fuse-med-ml},
  publisher = {Zenodo},
  year = {2021},
  copyright = {Apache License 2.0}
}
```
