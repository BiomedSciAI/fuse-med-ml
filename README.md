[![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg)](https://opensource.org/)
[![PyPI version](https://badge.fury.io/py/fuse-med-ml.svg)](https://badge.fury.io/py/fuse-med-ml)
[![Slack channel](https://img.shields.io/badge/support-slack-slack.svg?logo=slack)](https://join.slack.com/t/fusemedml/shared_invite/zt-xr1jaj29-h7IMsSc0Lq4qpVNxW97Phw)
[![Downloads](https://pepy.tech/badge/fuse-med-ml)](https://pepy.tech/project/fuse-med-ml)

<img src="fuse/doc/FuseMedML-logo.png" alt="drawing" width="30%"/>

# What is FuseMedML?
FuseMedML is an open-source python-based framework designed to enhance collaboration and accelerate discoveries in **F**used **M**edical data through advanced **M**achine **L**earning technologies. Initial version is PyTorch-based and focuses on deep learning on medical imaging.


# Why use FuseMedML?
Successful deep learning R&D must rely on knowledge and experiments, accumulated over a wide variety of projects, and developed by different people and teams.

FuseMedML is an outstanding collaboration framework that allows you to rerun an experiment or reuse some of the capabilities originally written for different projects—all with minimal effort.\
Using FuseMedML, you can write generic components that can be easily shared between projects in a plug & play manner, simplifying sharing and collaboration.

The framework’s unique software design provides many advantages, making it an ideal framework for deep-learning research and development in medical imaging:

* **Rapid development** -

* **Flexible, customizable, and scalable** -

* **Encourage sharing and collaboration** - 

* **Collection of off-the-shelf components and capabilities** - 

* **Standardized evaluation** - 

* **Medical imaging expertise** - 

* **Compatibility with alternative frameworks**

# Citation
If you use FuseMedML in scientific context, please consider citing us:
```bibtex
@misc{https://doi.org/10.5281/zenodo.5146491,
  doi = {10.5281/ZENODO.5146491},
  url = {https://zenodo.org/record/5146491},
  author = {IBM Research,  Haifa},
  title = {FuseMedML: https://github.com/IBM/fuse-med-ml},
  publisher = {Zenodo},
  year = {2021},
  copyright = {Apache License 2.0}
}
```
# Installation
The best way to install `FuseMedML` is to clone the repository and install it in an editable mode using `pip`:
```bash
$ pip install -e .
```
This mode, allows to edit the source code and easily contribute back to the open-source project.

An alternative, is to simply install using PyPI 
```bash
$ pip install fuse-med-ml
```
 
 FuseMedML supports Python 3.6 or later and PyTorch 1.5 or later. A full list of dependencies can be found in [**requirements.txt**](https://github.com/IBM/fuse-med-ml/tree/master/requirements.txt).
 

# Ready to get started?
## FuseMedML from the ground up
[**User Guide**](https://github.com/IBM/fuse-med-ml/tree/master/fuse/doc/user_guide.md) - including detailed explanation about FuseMedML modules, structure, concept, and more.

[**High Level Code Example**](https://github.com/IBM/fuse-med-ml/tree/master/fuse/doc/high_level_example.md) - example of binary classifier for mammography with an auxiliary segmentation loss and clinical data

## Examples
* classification
    * [**MNIST**](https://github.com/IBM/fuse-med-ml/tree/master/fuse_examples/classification/mnist/)  - a simple example, including training, inference and evaluation over [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
    * [**KNIGHT Challenge**](https://github.com/IBM/fuse-med-ml/tree/master/fuse_examples/classification/knight) - preoperative prediction of risk class for patients with renal masses identified in clinical Computed Tomography (CT) imaging of the kidneys. Including data pre-processing, baseline implementation and evaluation pipeline for the challenge.
    * [**Multimodality tutorial**](https://github.com/IBM/fuse-med-ml/blob/master/fuse_examples/tutorials/multimodality_image_clinical/multimodality_image_clinical.ipynb) - demonstration of two popular simple methods integrating imaging and clinical data (tabular) using FuseMedML  
    * [**Skin Lesion**](https://github.com/IBM/fuse-med-ml/tree/master/fuse_examples/classification/skin_lesion/) - skin lesion classification , including training, inference and evaluation over the public dataset introduced in [ISIC challenge](https://challenge.isic-archive.com/landing/2017)
    * [**Prostate Gleason Classifiaction**](https://github.com/IBM/fuse-med-ml/tree/master/fuse_examples/classification/prostate_x/) - lesions classification of Gleason score in prostate over the public dataset introduced in [SPIE-AAPM-NCI PROSTATEx challenge](https://wiki.cancerimagingarchive.net/display/Public/SPIE-AAPM-NCI+PROSTATEx+Challenges#23691656d4622c5ad5884bdb876d6d441994da38)
    * [**Lesion Stage Classification**](https://github.com/IBM/fuse-med-ml/tree/master/fuse_examples/classification/duke_breast_cancer/) - lesions classification of Tumor Stage (Size) in breast MRI over the public dataset introduced in [Dynamic contrast-enhanced magnetic resonance images of breast cancer patients with tumor locations (Duke-Breast-Cancer-MRI)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903)
    * [**Breast Cancer Lesion Classification**](https://github.com/IBM/fuse-med-ml/tree/master/fuse_examples/classification/MG_CMMD) - lesions classification of tumor ( benign, malignant) in breast mammography over the public dataset introduced in [The Chinese Mammography Database (CMMD)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230508)
    
## Walkthrough template
* [**Walkthrough Template**](https://github.com/IBM/fuse-med-ml/tree/master/fuse/templates/walkthrough_template.py) - includes several TODO notes, marking the minimal scope of code required to get your pipeline up and running. The template also includes useful explanations and tips.


## Evaluation package
The evaluation package of FuseMedML (fuse.eval) is a standalone library for evaluating ML models which not necessarily trained with FuseMedML.  
The package includes collection of off-the-shelf metrics and utilities such as calibration, thresholding, model comparison and more.
Details and examples can be found [here](https://github.com/IBM/fuse-med-ml/tree/master/fuse/eval/README.md)   

## Community support
We use the Slack workspace at fusemedml.slack.com for informal communication.
We encourage you to ask questions regarding FuseMedML that don't necessarily merit opening an issue on Github.

[**Use this invite link to join FuseMedML on Slack**](https://join.slack.com/t/fusemedml/shared_invite/zt-xr1jaj29-h7IMsSc0Lq4qpVNxW97Phw).

IBMers can also join a Slack channel in the IBM Research organization: 
[**#fusers**](https://ibm-research.slack.com/archives/C0176S37QNP) .


