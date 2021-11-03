# [ISBI 2022](https://biomedicalimaging.org/2022/) KNIGHT Challenge: Kidney clinical Notes and Imaging to Guide and Help personalize Treatment and biomarkers discovery

[**Challenge Website**]()

**Keywords: Radiology, KiTS, CT, Renal Cancer, Accelerated Discovery**

## Setup
The code shared for the challenge implemented using FuseMedML framework which requires simple installation.

[![Github repo](https://img.shields.io/static/v1?label=GitHub&message=FuseMedML&color=brightgreen)](https://github.com/IBM/fuse-med-ml)

[![PyPI version](https://badge.fury.io/py/fuse-med-ml.svg)](https://badge.fury.io/py/fuse-med-ml)

[![Slack channel](https://img.shields.io/badge/support-slack-slack.svg?logo=slack)](https://join.slack.com/t/fusemedml/shared_invite/zt-xr1jaj29-h7IMsSc0Lq4qpVNxW97Phw)

[![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg)](https://github.com/IBM/fuse-med-ml)


FuseMedML is an open-source python-based framework designed to enhance collaboration and accelerate discoveries in Fused Medical data through advanced Machine Learning technologies. 

Initial version is PyTorch-based and focuses on deep learning on medical imaging.

```
git clone https://github.com/IBM/fuse-med-ml.git
cd fuse-med-ml
pip install -e .
```

## Abstract
The aim of the KNIGHT challenge is to facilitate the development of Artificial Intelligence (AI) models for automatic preoperative prediction of risk class for patients with renal
masses identified in clinical Computed Tomography (CT) imaging of the kidneys. 

The dataset, we name the Kidney Classification (KiC) dataset, is based on the 2021 Kidney and Kidney Tumor Segmentation challenge (KiTS) and extended to include additional CT phases and clinical information, as well as risk classification labels, deducted from postoperative pathology results.


Some of the clinical information will also be available for inference. The patients are classified into five risk groups in accordance with American Urological Association (AUA) guidelines. 

These groups can be divided into two classes based on the follow-up treatment. 

The challenge consists of three tasks: (1) binary patient classification as per the follow-up treatment, (2) fine-grained classification into five risk groups and (3) discovery of prognostic biomarkers.

## Data
Kidney Classification (KiC) dataset. Details can be found in [challenge website]()

## Evaluation
The participants should submit a .csv file containing a row with class scores for each patient in the test set. The rows must adhere to the following scheme:

\[case_id,Task1-NoAT-score,Task1-CanAT-score,Task2-B-score,Task2-LR-score,Task2-IR-score,Task2-HR-score,Task2-VHR-score\]

Where “case_id" represents the sample (e.g. 00000) and all scores represent the probability of a patient to belong to a class.

The evaluation code together with a dummy prediction file can be found in `fuse-med-ml/fuse_examples/classification/knight/eval`
More details can be found in [challenge website]()

## Baseline Implementation
TBD