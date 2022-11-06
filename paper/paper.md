---
title: 'FuseMedML: a framework for accelerated discovery in machine learning based biomedicine'
tags:
  - Deep Learning
  - Machine Learning
  - Artificial Intelligence
  - Medical imaging
  - Clinical data
  - Computational biomedicine
authors:
  - name: Alex Golts
    corresponding: true # (This is how to denote the corresponding author)
    equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Moshiko Raboh
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Yoel Shoshan
    equal-contrib: true
    affiliation: 1
  - name: Sagi Polaczek
    affiliation: 1
  - name: Itai Guez
    affiliation: 1
  - name: Liam Hazan
    affiliation: 1
  - name: Efrat Hexter
    affiliation: 1
  - name: TBD
    affiliation: 1
  - name: TBD
    affiliation: 1
  
affiliations:
 - name: IBM Research - Israel
   index: 1
date: 26 October 2022
bibliography: paper.bib

---

# Summary

Machine Learning is at the forefront of scientific progress in Healthcare and Medicine. To accelerate scientific discovery, it is important to have tools that allow progress iterations to be collaborative, reproducible, reusable and easily built upon without "reinventing the wheel".
FuseMedML, or *fuse*, is a Python framework designed for accelerated Machine Learning (ML) based discovery in the medical domain. It is highly flexible and designed for easy collaboration, encouraging code reuse. Flexibility is enabled by a generic data object design where data is kept in a nested (hierarchical) Python dictionary (NDict), allowing to easily deal with information from different modalities. Functional components allow to specify input and output keys, to be read from and written to the nested dictionary.  
Easy code reuse is enabled through key components implemented as standalone packages under the main *fuse* repo using the same design principles. These include *fuse.data* - a flexible data processing pipeline, *fuse.dl* - reusable Deep Learning (DL) model architecture components and loss functions, and *fuse.eval* - a library for evaluating ML models.

# Statement of need
Medical related research projects span multiple modalities (e.g., imaging, clinical data, biochemical representations) and tasks (e.g., classification, segmentation, clinical conditions prediction).
Through experience with many such projects we found three key challenges:
1. Launching or implementing a new baseline model a new baseline model can take more time than it should. This is true even when very similar projects have already been done in the past by the same lab. 
2. Porting individual components across projects is often painful, resulting in researchers “reinventing the wheel” time after time.
3. Collaboration between projects across modalities as well as across domains such as imaging and molecules is very challenging.  

FuseMedML was designed with the goal of alleviating these challenged.

Before open sourcing it, we used *fuse* internally in multiple research projects [@raboh2022context], [@rabinovici2022early], [@rabinovici2022multimodal], [@jubran2021glimpse], [@tlusty2021pre], [@golts2022ensemble] and experienced significant improvement in development time, reusability and collaboration. 
We were also able to meaningfully measure our progress and statistical significance of our results with off-the-shelf *fuse.eval* components that facilitate metrics' confidence interval calculation and model comparison. These tools were enabled us to organize two challenges as part of the 2022 International Symposium on Biomedical Imaging (ISBI) [@knight], [@bright].

# Components

## *fuse.data*
FuseMedML's data package is designed for building a flexible data pipeline with reusable building blocks called *ops*. See \autoref{fig:diagram} for a simple example for how such a building block can be used across different projects.
Each op(eration) gets as an input a `sample_dict`, a dictionary that stores all the necessary information about a sample processed so far. Typically, an *op* also gets keys that specify what it should consider in `sample_dict` and where to store the output. Similarly, a minibatch is represented by a `batch_dict`.

A data pipeline consists of a `static_pipline` and a `dynamic_pipeline`. 
The output of the `static_pipeline` is cached to optimize running time and maximize GPU utilization.
The `dynamic_pipeline` is responsible for "online" processing that we don't want cached, such as random augmentations.
An instance of a *fuse* dataset class, which inherits from the PyTorch dataset class is then created from defined static and dynamic pipelines.  
The data package includes also includes generic utilities such as a PyTorch based sampler enabling batch class balancing and a tool for splitting data into folds acording to predefined criteria.

![In this example a medical image loader is the reusable *fuse* component. Different projects can have different formats for their data samples, but they can all use OpMedicalImageLoader by providing the appropriate key names when calling it. In Project B the same key name is used for the input and output, resulting in the loaded image data overriding the image paths in the updated sample.\label{fig:diagram}](figures/diagram.png)

## *fuse.dl*
FuseMedML's DL package works with PyTorch models, only modifying them to interact with a `batch_dict`. For training, *fuse.dl* utilizes PyTorch-Lightning, either through a ready made `LightningModuleDefault` class inheriting from Pytorch-Lightning's `LightningModule` class, or by allowing users who seek maximal customizability to implement their own custom `LightningModule` and operate in close resemblance to the standard PyTorch-Lightning workflow.    

## *fuse.eval*
The evaluation package of fuse is a standalone library for evaluating ML models for various performance metrics and comparing between models. It comes with advanced capabilities such as generic confidence interval wrapper for any metric, generic one vs. all extension for any binary metric to a multi-class scenario, metrics for model comparison taking into account statistical significance, model calibration tools, pipelining for combining a sequence of metrics with possible dependencies, automatic per-fold evaluation and sub-group analysis, automaric handling of massive data through batching and multiprocessing, and more. 


# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

TBD

# References
