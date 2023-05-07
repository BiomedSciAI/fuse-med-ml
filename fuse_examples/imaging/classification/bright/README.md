# [ISBI 2022](https://biomedicalimaging.org/2022/) BRIGHT Challenge: BReast tumor Image classification on Gigapixel HisTopathological images

[**Challenge Website**](https://research.ibm.com/haifa/Workshops/BRIGHT)

**Keywords: Computational Pathology, WSI-Classification, Atypias, Breast tumor Subtyping**

## Setup
See https://github.com/BiomedSciAI/fuse-med-ml#installation

## Abstract
The aim of the BRIGHT challenge is to provide an opportunity for the development, testing and evaluation of Artificial Intelligence (AI) models for automatic breast tumor subtyping of frequent lesions along with rare pathologies, by using clinical Hematoxylin & Eosin (H&E) stained gigapixel Whole-Slide Images (WSIs).

To this end, a large annotated cohort of WSIs, which includes Noncancerous (Pathological Benign, Usual Ductal Hyperplasia), Precancerous (Flat Epithelia Atypia, Atypical Ductal Hyperplasia) and Cancerous (Ductal Carcinoma in Situ, Invasive Carcinoma) categories, will be available. BRIGHT is the first breast tumor subtyping challenge that includes atypical lesions and consists of more than 550 annotated WSIs across a wide spectrum of tumor subtypes.

The Challenge includes two tasks: (1) WSI classification into three classes as per cancer risk, and (b) WSI classification into six fine-grained lesion subtypes.

## Data
BReAst Carcinoma Subtyping (BRACS) dataset , a cohort of H&E-stained breast tissue biopsies.. Details can be found in [challenge website](https://research.ibm.com/haifa/Workshops/BRIGHT)

## Evaluation
The participants should submit a .csv file per task containing a row with a final class predictions and per-class score for each patient in the test set. The rows must adhere (including header row) to the following scheme:

**Task 1 Prediction File:**
\[image_name,predicted_label,Noncancerous-score,Precancerous-score,Cancerous-score\]

See [example prediction file for task 1](./eval/example/example_task1_predictions.csv)

**Task 2 Prediction File:**
\[image_name,predicted_label,PB-score,UDH-score,FEA-score,ADH-score,DCIS-score,IC-score\]

See [example prediction file for task 2](./eval/example/example_task2_predictions.csv)

Where “image_name" represents the sample (e.g. BRACS_264) and all scores represent the probability of a patient to belong to a class.

The evaluation script together with a dummy prediction files can be found in `fuse_examples/imaging/classification/bright/eval`
More details can be found in [challenge website](https://research.ibm.com/haifa/Workshops/BRIGHT)


<br/>

To run the evaluation script:
```
cd fuse_examples/imaging/classification/knight/eval
python eval.py <target_filename> <task1 prediction_filename> <task1 prediction_filename> <output dir>
```
To evaluate the dummy example predictions and targets
```
cd fuse_examples/imaging/classification/knight/eval
python eval.py example/example_targets.csv example/example_task1_predictions.csv example/example_task2_predictions.csv example/results
```

### Baseline
As an additional example, we also include the validation prediction files and validation target file of the challenge baseline implementation:


See [validation baseline prediction file for task 1](./eval/baseline/validation_baseline_task1_predictions.csv)


See [validation baseline prediction file for task 2](./eval/baseline/validation_baseline_task2_predictions.csv)


See [validation targets file](./eval/validation_targets.csv)



<br/>

To evaluate the baseline predictions over the validation set:
```
cd fuse_examples/imaging/classification/bright/eval
python eval.py validation_targets.csv baseline/validation_baseline_task1_predictions.csv baseline/validation_baseline_task2_predictions.csv baseline/validation_results
```
