# Virtual Biopsy Derived Using AI-based Multimodal Modeling of Binational Breast Mammography Data

## About

This repository is an implementation of Virtual Biopsy, an AI-based algorithm trained on digital mammography (DM) and linked electronic health records to classify biopsied breast lesions. In the paper, a model that combines convolutional neural networks (CNN) with classic supervised learning algorithms was independently trained on DM from 2,120/1,642 women in Israel and the USA to predict breast lesion malignancy, obtaining an AUC on the held-out sets of 0.88 (0.85, 0.91) and 0.80 (0.74, 0.85), respectively. In the breast lesion subtype prediction, the best performing algorithms obtained AUC of 0.85 (0.82, 0.89) for ductal carcinoma in situ (DCIS), 0.76 (0.72, 0.83) for invasive leasions and 0.82 (0.77, 0.87) for benign lesions.

Since the DM datasets from the original paper are not publicly available, in this repository we provide the exact implementation details of Virtual Biopsy using a different dataset called [KNIGHT](https://github.com/neheller/KNIGHT). Although KNIGHT consists of Computed Tomography (CT) imaging of the kidneys, it includes additional rich clinical information of the patients as well as risk classification labels deduced from postoperative pathology results. Thus, it is a favorable dataset to exemplify two basic operations in the Virtual Biopsy workflow: (i) integration of imaging with clinical features and (ii) multiclass classification. In the later, instead of breast lesion histopathology classification into DCIS, invasive, benign, high-risk lesions or 'others', we classify patients according to their risk categories: benign, low-risk, intermediate-risk, high-risk and very-high-risk. The approach used with the KNIGHT dataset was the same used for the MG dataset, and we illustrate the workflow below on the jupyter notebook ```virtual_biopsy_workflow.ipynb```.


## Installing Dependencies

Virtual Biopsy is based on [FuseMedML](https://github.com/IBM/fuse-med-ml) framework and is tested on Python >= 3.7 (3.7 is recommended) and PyTorch >= 1.5.

- Install [Conda](https://www.anaconda.com/blog/moving-conda-environments)
- Create a conda environment using the following command (you can replace FUSEMEDML with your preferred environment name)

```
    conda create -n FUSEMEDML python=3.7
    conda activate FUSEMEDML
```

- Install from source in an [editable mode](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs) using ```pip```:

```
pip install -e .
```

## Download the data

- Clone the official [KNIGHT database repository](https://github.com/neheller/KNIGHT).
- A JSON file with each patient's clinical data is located under `knight/data/knight.json`. The imaging associated with each of the 300 patients needs to be downloaded with the `knight/scripts/get_imaging.py` script (requires Python 3).
- ```cd``` into ```KNIGHT``` and run ```python knight/scripts/get_imaging.py``` to download the images.

## About the data

The prediction target in the KNIGHT data is the attribute entitled `"aua_risk_group"` in the `knight.json` file. The primary task is a binary classification between the two higher-risk groups (`"high_risk"` and `"very_high_risk"`, the patients who are candidate for adjuvant therapy) versus the three lower-risk groups (`"benign"`, `"low_risk"`, and `"intermediate_risk"`, consisting of patients who do not need adjuvant therapy). This label is somehow similar to our original DM dataset, where we perform classification between two malignant breast lesion classes (`"DCIS"` and "`invasive leasions"`) versus the three benign breast lesion groups ("`benign"`, "`high-risk lesions"` and "`others"`). Thus, task 1 is referred as "`cancer vs. non-cancer"` classification in our paper.
A secondary task is the five-way classification problem for each group individually, which represents the breast lesion subtypes classification in the paper.

## Steps to run the code

1. Once you cloned the KNIGHT database repository (which should now contain the imaging and clinical data under ```knight/data```), you need to update the ```data_path``` variable inside the ```CNN_example.py``` script to be your local path where the images are stored (e.g. ```~/VirtualBiopsy/KNIGHT```)
2. Now you can run the the ```CNN_example.py``` script which trains and evaluates the CNN model. In the top of the script, are some parameters that you can change:

(todo: explain parameters) XXXXX
