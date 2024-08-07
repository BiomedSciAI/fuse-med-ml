# 3D Medical Imaging Pre-training and Downstream Task Validation

This example provides code for 3D medical imaging pre-training and validation on downstream tasks such as classification and segmentation. It implements self-supervised learning techniques, specifically DINO and allows for fine-tuning on various medical imaging tasks.

## Data Preparation

For each training type (self-supervised, classification, segmentation), prepare a CSV file with the following structure:

### Self-Supervised and Classification CSV

| PatientID | path | fold |
|-----------|------|------|
| ID1       | /path/to/dicom/folder1 | 0 |
| ID2       | /path/to/dicom/folder2 | 1 |

For classification, add an additional column for the category to predict (as defined in `classification/config.yaml`).

### Segmentation CSV

| PatientID | img_path | seg_path | fold |
|-----------|----------|----------|------|
| ID1       | /path/to/image1.nii.gz | /path/to/segmentation1.nii.gz | 0 |
| ID2       | /path/to/image2.nii.gz | /path/to/segmentation2.nii.gz | 1 |

The "fold" column is used for cross-validation and should contain integer values representing different folds.

## Configuration

Each training type has its own `config.yaml` file. Make sure to set the following parameters:

- `results_dir`: Path to save results and checkpoints
- `csv_path`: Path to the CSV file for the respective training type
- `experiment`: Name of the experiment and also the name of the results folder
- `train_folds`: List of fold values to use for training (e.g., [0, 1, 2])
- `val_folds`: List of fold values to use for validation (e.g., [3])
- `test_folds`: List of fold values to use for testing (e.g., [4])
- `test_ckpt`: Path to the checkpoint for testing. If set to "null", the model will train using the train and validation sets. If a path is provided, it will perform evaluation on the test set using the given checkpoint.

If using pretrained weights (`pretrained: true`), also set:
- `pretrained_path`: Path to the pretrained weights

Pretrained weights can be downloaded from [SuPreM GitHub repository](https://github.com/MrGiovanni/SuPreM).

## Running the Training
