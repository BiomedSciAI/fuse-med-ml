[![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg)](https://opensource.org/)
[![PyPI version](https://badge.fury.io/py/fuse-med-ml.svg)](https://badge.fury.io/py/fuse-med-ml)
[![Slack channel](https://img.shields.io/badge/support-slack-slack.svg?logo=slack)](https://join.slack.com/t/newworkspace-i3g4445/shared_invite/zt-sr0hcb9f-E~SLYbG9bE5fn8iq5OE0ww)

<img src="fuse/doc/FuseMedML-logo.png" alt="drawing" width="30%"/>

# What is FuseMedML?
FuseMedML is an **open-source PyTorch-based framework designed to enhance collaboration and to facilitate deep learning R&D in medical imaging**.
# Why use FuseMedML?
Successful deep learning R&D must rely on knowledge and experiments accumulated on a wide variety of projects developed by different people and teams. Even though many of those experiments in the medical field shares lots in common, collaboration and code reuse are challenging tasks.

FuseMedML is a great collaboration framework that will allow to rerun an experiment or reuse some of the capabilities originally written for a different project with minimum effort. 
Using FuseMedML, you will be able to write generic components which are decoupled from other components, the structure of the data, and the model output.  This kind of component can be easily shared between projects in a plug & play manner and therefore encourage sharing and collaboration significantly.

A software design achieving those goals, provides many advantages, making it, in general, a great framework for deep-learning research and development in medical imaging:
* **Rapid development** -

  Requires implementing only minimal scope of code to get up and running fully-featured pipeline, including caching, augmentation, monitoring, logging, and many more. 
  Common generic implementation of most components in the pipeline is provided and only specific components such as data extractor are expected to be implemented by the user.
* **Flexible, customizable, and scalable** -

  The default implementation of modules and components suits many common cases. 
  However, components in the pipeline kept decoupled, allowing a user to re-implement a component to get the required behavior.
* **Encourage sharing and collaboration** - 

  Sharing and collobration are powerful tools and therefore they are FuseMedML's main goals.
* **Collection of common, easy to use, generic components and capabilities** - 

  FuseMedML comes with a large collection of components that will grow with each new project.
  Examples: Monitoring, crash recovery, caching, augmentation, visualization, data sampling/balancing, metrics, losses, etc.
* **Standardized evaluation** - 

  The evaluation methods and code are completely shared and therefore contribute to a standardized evaluation.
* **Medical Imaging Expertise** - 

  The pre-implemented components implemented by a group of medical-imaging researchers and tackle many of the challenges in this field.   
* **Comaptibilty with alternative frameworks**

    Most code components developed in alternative frameworks are compatible with FuseMedML and can still be used.
    
    Many other popular GitHub projects, such as the implementation of a PyTorch model, can be used as complementary projects to FuseMedML
# Installation
The best way of installing `FuseMedML` is by using `pip`:
```bash
$ pip install fuse-med-ml
```
 FuseMedML supports Python 3.6 or later and PyTorch 1.5 or later. Full dependencies list can be found in [**requirements.txt**](requirements.txt).
 
An alternative, allowing to edit the source code, would be to download the repo and install it using:
```bash
$ pip install -e .
```

# Ready to get going?
## FuseMedML from the ground up
[**User Guide**](fuse/doc/user_guide.md) - including detailed explanation about FuseMedML modules, structure, concept. etc.

[**High Level Code Example**](fuse/doc/high_level_example.md) - example for binary classifier for mammography with an auxiliary segmentation loss and clinical data

## Examples
* classification
    * [**MNIST**](fuse_examples/classification/mnist/)  - a simple example, including training, inference and evaluation over [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
    * [**Skin Lesion**](fuse_examples/classification/skin_lesion/) - skin lesion classification , including training, inference and evaluation over the public dataset introduced in [ISIC challenge](https://challenge.isic-archive.com/landing/2017)
    * [**Prostate Gleason Classifiaction**](fuse_examples/classification/prostate_x/) - lesions classification of Gleason score in prostate over the public dataset introduced in [SPIE-AAPM-NCI PROSTATEx challenge](https://wiki.cancerimagingarchive.net/display/Public/SPIE-AAPM-NCI+PROSTATEx+Challenges#23691656d4622c5ad5884bdb876d6d441994da38)

## Walkthrough template
* [**Walkthrough Template**](fuse/templates/walkthrough_template.py) - the template includes several TODO notes, marking the minimal scope of code required to get up and running pipeline. The template also includes useful explanations and tips.

## Community support
We use the Slack workspace at fusemedml.slack.com for informal communication.
We encourage you to ask questions regarding FuseMedML that don't necessarily merit opening an issue on Github.

[**Use this invite link to join FuseMedML on Slack**](https://join.slack.com/t/newworkspace-i3g4445/shared_invite/zt-sr0hcb9f-E~SLYbG9bE5fn8iq5OE0ww).

IBMers can also join a slack channel on IBM Research organization:
[**#fusers**](https://ibm-research.slack.com/archives/C0176S37QNP) .


