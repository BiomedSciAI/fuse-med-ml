# [ISBI 2022](https://biomedicalimaging.org/2022/) KNIGHT Challenge: Kidney clinical Notes and Imaging to Guide and Help personalize Treatment and biomarkers discovery

[**Challenge Website**](https://research.ibm.com/haifa/Workshops/KNIGHT)

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
The participants should submit a .csv file per task containing a row with class scores for each patient in the test set. The rows must adhere to the following scheme:

**Task 1 Prediction File:**
\[case_id,NoAT-score,CanAT-score\]

See [example prediction file for task 1](https://github.com/IBM/fuse-med-ml/blob/master/fuse_examples/classification/knight/eval/example/example_task1_predictions.csv)

**Task 2 Prediction File:**
\[case_id,B-score,LR-score,IR-score,HR-score,VHR-score\]

See [example prediction file for task 2](https://github.com/IBM/fuse-med-ml/blob/master/fuse_examples/classification/knight/eval/example/example_task2_predictions.csv)

Where “case_id" represents the sample (e.g. 00000) and all scores represent the probability of a patient to belong to a class.

The evaluation script together with a dummy prediction file can be found in `fuse-med-ml/fuse_examples/classification/knight/eval`

More details can be found in [challenge website](https://research.ibm.com/haifa/Workshops/KNIGHT)

<br/>

To run the evaluation script:

```
cd fuse-med-ml/fuse_examples/classification/knight/eval
python eval.py <target_filename> <task1 prediction_filename> <task1 prediction_filename> <output dir>
```
To evaluate the dummy example predictions and targets
```
cd fuse-med-ml/fuse_examples/classification/knight/eval 
python eval.py example/example_targets.csv example/example_task1_predictions.csv example/example_task2_predictions.csv example/results
```    

## KNIGHT FuseMedML Baseline Implementation
This example demonstrates a basic classification pipeline for the KNIGHT challenge Kidney Classification (KiC) dataset. We utilize the FuseMedML library for all stages in the pipeline including data preprocessing, feature extraction, network training and metrics calculation.
In this example, we extract features from the CT volumes using a 3D ResNet-18 backbone, concatenate them with available clinical features processed by a smaller fully connected network, and classify the combined features into two risk groups as defined in the 1st task of the KNIGHT challenge. 

### Steps to run the code
0. As a preliminary step, make sure you install [FuseMedML](https://github.com/IBM/fuse-med-ml) according to the [installation instructions](https://github.com/IBM/fuse-med-ml#installation). In order to understand the basics of the library in high level, please read as a minimum this [user guide](https://github.com/IBM/fuse-med-ml/blob/master/fuse/doc/user_guide.md).  
You can also access more documentation including code examples and tutorials from the [FuseMedML Github page](https://github.com/IBM/fuse-med-ml).
1. Download the KNIGHT database from the official [KNIGHT database repository](https://github.com/neheller/KNIGHT). The KNIGHT database is strongly based on the [KiTS21 database](https://github.com/neheller/kits21). In KiTS21 the task was semantic segmentation, and there are segmentation annotations for it. You might want to benefit from them in your approach to the KNIGHT challenge and you may access them by cloning the official [KiTS21 database repository](https://github.com/neheller/kits21). But you won't need them for this example.  
2. ```cd``` into ```KNIGHT``` and run ```python knight/scripts/get_imaging.py``` to download the imaging.
3. Set the environment variable ```KNIGHT_DATA``` to your local path to the cloned KNIGHT database repository (which should now contain the imaging and clinical data under ```knight/data```).  
Additionally, set the environment variable ```KNIGHT_CACHE``` to the path where you want to store cached data. Lastly, set the ```KNIGHT_RESULTS``` environment variable to the location where you want to store your trained models.

4. Now you can run the ```baseline/fuse_baseline.py``` script which trains and evaluates our baseline model. In the top of the script, are some parameters that you can change.  
The ```use_data``` dictionary sets whether to use only clinical data, only imaging, or both. We encourage you to develop a solution which utilizes and benefits from both. The "clinical only" mode trains much faster and requires significantly less resources, as expected.   
The ```resize_to``` parameter sets a common size to which we resize the input volumes.  
**Important Note:** Once you run ```fuse_baseline.py``` for the first time, it will perform caching of the data and store the cached data in your ```KNIGHT_CACHE``` folder. The next time you run ```fuse_baseline.py```, it will skip this process, and use the existing, already cached data. You need to be aware of this, because if you want to modify anything related to the data (for example, the ```resize_to``` parameter), then you will need to manually delete the contents of ```KNIGHT_CACHE``` folder, to allow the caching process to take place again with the new parameters.

5. While running the ```fuse_baseline.py``` script, you can monitor your model training using an instance of Tensor Board, for which the ```logdir``` param is set to your ```KNIGHT_RESULTS``` directory.  

### Implementation details
In this example we perform some basic preprocessing on the imaging data. We downsize the images by a factor of 2 in the X and Y dimensions (to 256), which is only meant for efficiency and not necessarily recommended. Additionally we resize in the Z dimension to 110, the median number of slices. We clip the voxel values to [-62, 310] (corresponds to 0.5 and 99.5 percentile in the foreground regions). Then, we subtract 104.9 and divide by 75.3 (corresponds to mean and std in the foreground regions, respectively). We also normalize the clinical features according to their approximate expected maximum value.   

We used two 32GB V-100 GPUs for running this baseline and reporting the results below. 
However, we did manage to train an "Imaging + Clinical" model with comparable results on a single 12GB GPU by reducing the batch size to 2 and changing the ```resize_to``` parameter to ```(256, 256, 64)```.

### Results
Running the code with the parameters currently set in ```fuse_baseline.py``` gave the following best epoch validation AUC results:

Clinical data only | Imaging only | Imaging + Clinical
:---:|:---:| :---:
0.87 | 0.71 | 0.85

### Limitations
Our goal in this example is to provide easy to follow code that can help you get started with the KNIGHT challenge using FuseMedML. We encourage you to experiment with, improve, or change completely any step in the proposed pipeline. We also encourage you to learn about [solutions to the KiTS19 and KiTS21 challenges](https://kits21.kits-challenge.org/results), to gain useful insight into working with this imaging data, towards a related, but different task. 

Here are some of the things we knowingly avoided for the sake of simplicity:
1. We use most but not all of the available clinical features, and we simplify some of those that we do use. For example, we use a simplified binary version of the "comorbidities" feature which only cares whether or not a patient has any comorbidity. In practice the data contains more specific information, which you may want to use. 

2. We didn't make any use of the segmentation labels that are given in the KiTS database. You can use them in any way to improve your preprocessing, data sampling or model training. Note that the segmentations won't be available for the test cases.

3. We didn't resample the images with respect to their spacing, but only resized to a common voxel size. Addressing the trade-off between input patch size (limited by the GPU memory) and the amount of contextual information that it contains (controlled by a possible resampling procedure) can be important. You may want to resample the volumes to a common spacing, and you may want (or be forced to, due to GPU memory constraints), to train on smaller cropped patches, with some logic which "prefers" foreground patches.

### **Make targets file for evaluation**
'fuse-med-ml/fuse_examples/classification/knight/make_targets_file.py' is a script that makes a targets file for the evaluation script. 

Targets file is a csv file that holds just the labels for both tasks. This files is one of the inputs of the evaluation script.

The script extracts the labels from the PyTorch dataset included in baseline implementation.

The baseline implementation is using specific train/validation split, You can either use the same train/validation split or set a different split.

The script including additional details and documentation can be found in: 'fuse-med-ml/fuse_examples/classification/knight/make_targets_file.py'  

### **Make predictions file for evaluation**
'fuse-med-ml/fuse_examples/classification/knight/make_predictions_file.py' is a script that automatically makes predictions files for any model trained using FuseMedML.

Predictions file is a csv file that include prediction score per class and should adhere a format specified in evaluation section.

The script including additional details and documentation can be found in: 'fuse-med-ml/fuse_examples/classification/knight/make_predictions_file.py'  


