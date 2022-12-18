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
  - name: Moshe Raboh
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Yoel Shoshan
    equal-contrib: true
    affiliation: 1
  - name: Sagi Polaczek
    affiliation: 1
  - name: Simona Rabinovici-Cohen
    affiliation: 1    
  - name: Efrat Hexter
    affiliation: 1
  
affiliations:
 - name: IBM Research - Haifa, Israel
   index: 1
date: 26 October 2022
bibliography: paper.bib

---

# Summary

Machine Learning is at the forefront of scientific progress in Healthcare and Medicine. To accelerate scientific discovery, it is important to have tools that allow progress iterations to be collaborative, reproducible, reusable and easily built upon without "reinventing the wheel".  
FuseMedML, or *fuse*, is a Python framework designed for accelerated Machine Learning (ML) based discovery in the medical domain. It is highly flexible and designed for easy collaboration, encouraging code reuse. Flexibility is enabled by a generic data object design where data is kept in a nested (hierarchical) Python dictionary (NDict), allowing to efficiently process and fuse information from multiple modalities. Functional components allow to specify input and output keys, to be read from and written to the nested dictionary.  
Easy code reuse is enabled through key components implemented as standalone packages under the main *fuse* repo using the same design principles. These include *fuse.data* - a flexible data processing pipeline, *fuse.dl* - reusable Deep Learning (DL) model architecture components and loss functions, and *fuse.eval* - a library for evaluating ML models.  

# Statement of need
Medical related research projects span multiple modalities (e.g., imaging, clinical data, biochemical representations) and tasks (e.g., classification, segmentation, clinical conditions prediction).
Through experience with many such projects we found three key challenges:
1. Launching or implementing a new baseline model can take more time than it should. This is true even when very similar projects have already been done in the past by the same lab. 
2. Porting individual components across projects is often painful, resulting in researchers “reinventing the wheel” time after time.
3. Collaboration between projects across modalities as well as across domains such as imaging and molecules is very challenging.  

FuseMedML was designed with the goal of alleviating these challenges.  

Before open sourcing it, we used *fuse* internally in multiple research projects [@raboh2022context], [@rabinovici2022early], [@rabinovici2022multimodal], [@jubran2021glimpse], [@tlusty2021pre], [@golts2022ensemble], [@radiology] and experienced significant improvement in development time, reusability and collaboration. 
We were also able to meaningfully measure our progress and statistical significance of our results with off-the-shelf *fuse.eval* components that facilitate metrics' confidence interval calculation and model comparison. These tools have enabled us to organize two challenges as part of the 2022 International Symposium on Biomedical Imaging (ISBI) [@knight], [@bright].

# State of the field
FuseMedML is a comprehensive ML library with an emphasis on the biomedical domain. It provides a broad set of tools spanning the whole development pipeline, from data preparation, through model training to evaluation. It is built on top of leading ML frameworks such as PyTorch [@NEURIPS2019_bdbca288] and PyTorch Lightning [@Falcon_PyTorch_Lightning_2019], and attempts to complement them where needed, as well as introduce domain specific layers.  
One way in which *fuse* is unique is in it's flexible design concept of storing data in a specialized nested dictionary. This is a key driver of flexibility, allowing minimal code modifications when moving building blocks between different projects.  
There are existing PyTorch-based ML libraries that similarly to *fuse* cater to researchers in the biomedical domain. Two such prominent libraries are MONAI [@Cardoso_MONAI_An_open-source_2022] and PyHealth [@DBLP:journals/corr/abs-2101-04209]. One aspect in which *fuse* may complement MONAI is in the availability of tools for comparing between models as well as analyzing statistical significance of metric results. In PyHealth, unlike *fuse* the main focus appears to be on health records data, with not enough emphasis on medical imaging data and models. 

# Packages

## *fuse.data*
FuseMedML's data package is designed for building a flexible and powerful data pipeline with reusable building blocks called *ops*. See \autoref{fig:diagram} for a simple example for how such a building block can be used across different projects.  
Each *op* class's `__call__` function gets as an input a `sample_dict`, a dictionary that stores all the necessary information about a sample processed so far. Typically, an *op*'s constructor gets keys that specify what it should consider in `sample_dict` and where to store the output. Similarly, a minibatch is represented by a `batch_dict`.  
A special kind of *ops* are "Meta *ops*". They can be thought of as a form of wrapper *op* around a regular, lower level *op* or function, to help achieve a special behavior such as repeating that low level *op*, applying it with random values and more. "Meta *ops*" also help avoid writing boilerplate code.  

A data pipeline may consist of a `static_pipline` and a `dynamic_pipeline`. 
The output of the `static_pipeline` can be cached to optimize running time and GPU utilization.
The `dynamic_pipeline` is responsible for "online" processing that we don't want to cache, such as random augmentations.
An instance of a *fuse* dataset class, which inherits from the PyTorch dataset class is then created from defined static and dynamic pipelines.  
The data package also includes generic utilities such as a PyTorch based sampler enabling batch class balancing and a tool for splitting data into folds acording to predefined criteria.

![In this example a medical image loader is the *fuse* component reused in projects A and B. Different projects can have different formats for their data samples, but they can all use OpMedicalImageLoader by providing the appropriate key names when calling it. In Project B the same key name is used for the input and output, resulting in the loaded image data overriding the image paths in the updated sample.\label{fig:diagram}](figures/diagram.png){width=100%}

## *fuse.dl*
FuseMedML's DL package works with PyTorch models, only modifying them to interact with a `batch_dict`. For training, *fuse.dl* utilizes PyTorch-Lightning, either through an already made `LightningModuleDefault` class that inherits from Pytorch-Lightning's `LightningModule` class, or by allowing users who seek maximal customizability to implement their own custom `LightningModule` and operate in close resemblance to the standard PyTorch-Lightning workflow or use alternative training loop implementations.   

## *fuse.eval*
The evaluation package of fuse is a standalone library for evaluating ML models for various performance metrics and comparing between models. It comes with advanced capabilities such as generic confidence interval wrapper for any metric, generic one vs. all extension for any binary metric to a multi-class scenario, metrics for model comparison taking into account statistical significance, model calibration tools, pipelining for combining a sequence of metrics with possible dependencies, automatic per-fold evaluation and sub-group analysis, automaric handling of massive data through batching and multiprocessing, and more. See \autoref{fig:metric_pipeline} for an example of an evaluation metric pipeline that can be reused across projects.  

![In this example a pipeline of evaluation metric components is shown. It consists of two metrics: the Area Under the receiver operating characteristic Curve and the Area Under the Precision-Recall Curve. Both metrics are wrapped with a Confidence Interval (CI) metric, resulting in a lower and upper bound for each metric. \label{fig:metric_pipeline}](figures/metric_pipeline.png){width=100%}

# Extensions
The core technology of *fuse* and its component packages is general. More specific functionality which involves domain expertise is contained within extensions. One such extension, *fuse-imaging* is currently published. It extends the *fuse.data* package with *ops* useful for medical imaging, as well as public medical dataset implementations.  

# References
