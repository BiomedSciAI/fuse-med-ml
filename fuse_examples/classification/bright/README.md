# [ISBI 2022](https://biomedicalimaging.org/2022/) BRIGHT Challenge: BReast tumor Image classification on Gigapixel HisTopathological images

[**Challenge Website**]()

**Keywords: Computational Pathology, WSI-Classification, Atypias, Breast tumor Subtyping**

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
The aim of the BRIGHT challenge is to provide an opportunity for the development, testing and evaluation of Artificial Intelligence (AI) models for automatic breast tumor subtyping of frequent lesions along with rare pathologies, by using clinical Hematoxylin & Eosin (H&E) stained gigapixel Whole-Slide Images (WSIs). 

To this end, a large annotated cohort of WSIs, which includes Noncancerous (Pathological Benign, Usual Ductal Hyperplasia), Precancerous (Flat Epithelia Atypia, Atypical Ductal Hyperplasia) and Cancerous (Ductal Carcinoma in Situ, Invasive Carcinoma) categories, will be available. BRIGHT is the first breast tumor subtyping challenge that includes atypical lesions and consists of more than 550 annotated WSIs across a wide spectrum of tumor subtypes. 

The Challenge includes two tasks: (1) WSI classification into three classes as per cancer risk, and (b) WSI classification into six fine-grained lesion subtypes.

## Data
BReAst Carcinoma Subtyping (BRACS) dataset , a cohort of H&E-stained breast tissue biopsies.. Details can be found in [challenge website]()

## Evaluation
The participants should submit a .csv file per task containing a row with a final class predictions and per-class score for each patient in the test set. The rows must adhere (including header row) to the following scheme:

**Task 1 Prediction File:**
\[image_name,predicted_label,Noncancerous-score,Precancerous-score,Cancerous-score\]

See [example prediction file]("[eval/example_task1_predictions.csv](https://github.com/IBM/fuse-med-ml/blob/master/fuse_examples/classification/bright/eval/example/example_task1_predictions.csv)")

**Task 2 Prediction File:**
\[image_name,predicted_label,PB-score,UDH-score,FEA-score,ADH-score,DCIS-score,IC-score\]

See [example prediction file]("https://github.com/IBM/fuse-med-ml/blob/master/fuse_examples/classification/bright/eval/example/example_task2_predictions.csv")

Where “image_name" represents the sample (e.g. BRACS_264) and all scores represent the probability of a patient to belong to a class.

The evaluation script together with a dummy prediction files can be found in `fuse-med-ml/fuse_examples/classification/bright/eval`
More details can be found in [challenge website]()

### Baseline
As an additional example, we also include the validation prediction files and validation target file of the challenge baseline implementation:
 
See [validation baseline prediction file for task 1]("https://github.com/IBM/fuse-med-ml/blob/master/fuse_examples/classification/bright/eval/baseline/validation_baseline_task1_predictions.csv")

See [validation baseline prediction file for task 2]("https://github.com/IBM/fuse-med-ml/blob/master/fuse_examples/classification/bright/eval/baseline/validation_baseline_task2_predictions.csv)

See [validation targets file]("https://github.com/IBM/fuse-med-ml/blob/master/fuse_examples/classification/bright/eval/validation_targets.csv")