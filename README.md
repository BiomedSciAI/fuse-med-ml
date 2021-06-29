[![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg)](https://opensource.org/)
[![Build Status](https://travis-ci.org/IBM/FuseMedML.svg?&branch=master)](https://travis-ci.org/IBM/FuseMedML)
[![PyPI version](https://badge.fury.io/py/FuseMedML.svg)](https://badge.fury.io/py/FuseMedML)
[![Slack channel](https://img.shields.io/badge/support-slack-slack.svg?logo=slack)](https://ibm-research.slack.com/archives/C0176S37QNP)

![Fuse Logo](Fuse/doc/FuseMedML-logo.png "Fuse")

# What is FuseMedML?
FuseMedML is an **open-source PyTorch based framework designed to facilitate deep learning R&D in medical imaging**.
FuseMedML provides entire built-in pipeline, from data processing through train and inference to analyze and visualize.

# Why use FuseMedML?
FuseMedML punctilious software design provides many advantages, making it a great framework for deep-learning research and development in medical imaging:  
* **Rapid Development** -
  Requires implementing only minimal scope of code to get up and running pipeline.
  Common generic implementation of most components in pipeline is provided and only specific components such as data extractor expected to be implemented by the user.
* **Flexible, customizable, and scalable** -
  The default implementation of modules and components suits for many common cases, 
  However, components in the pipeline kept decoupled, allowing a user to re-implement a component to get the required behavior.
* **Encourage sharing** -   
  Sharing is a powerful tool, To encourage sharing, FuseMedML design enables to implement a generic components which can be easily used in Plug & Play manner.
* **Common easy to use generic components and examples** - 
  Such as: Caching, Augmentor, Visualizer, Sampler/Balancer, Metrics, Losses, etc.
* **Medical Imaging Expertise** - 
  The pre-implemented components implemented by a group of medical-imaging researchers and tackles many of the challenges in this field.   

# Installation
The best way of installing `FuseMedML` is by using `pip`:
```bash
$ pip install fuse_med_ml
```
 FuseMedML supports Python 3.6 or later and PyTorch 1.5 or later. Full dependencies list can be found in [**requirements.txt**](requirements.txt).
 
An alternative, allowing to edit the source code, would be to download the repo and install it using:
```bash
$ pip install -e .
```

# Ready to get going?
## FuseMedML from the ground up
[**User Guide**](Fuse/doc/user_guide.md) - including detailed explanation about FuseMedML modules, structure, concept. etc.

[**High Level Code Example**](Fuse/doc/high_level_example.md) - example for binary classifier for mammography with an auxiliary segmentation loss and clinical data

## Examples
* classification
    * A simple example, including training, inference and evaluation over [MNIST](http://yann.lecun.com/exdb/mnist/)  dataset: [**classification_mnist**](Fuse/examples/classification/mnist/) 

## Walkthrough template
* [**Walkthrough Template**](Fuse/templates/walkthrough_template.py) - the template includes several TODO notes, marking the minimal scope of code required to get up and running pipeline. The template also includes useful explanation and tips.

## Support
**#fusers** Slack channel on IBM Research organization.


