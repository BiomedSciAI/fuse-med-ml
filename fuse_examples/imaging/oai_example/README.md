# Project Name

Brief description of your project.

## Project Structure

│
├── data/
│   ├── data_ops.py
│   ├── oai_ds.py
│   └── seg_ds.py
│
├── downstream/
│   ├── classification.yaml
│   ├── classification.py
│   ├── segmentation.yaml
│   └── segmentation.py
│
├── self_supervised/
│   ├── dino.yaml
│   └── dino.py



## Data Preparation

For each training type (self-supervised, classification, segmentation), prepare a CSV file with the following structure:

### Self-Supervised and Classification CSV

| PatientID | path |
|-----------|------|
| ID1       | /path/to/dicom/folder1 |
| ID2       | /path/to/dicom/folder2 |

For classification, add an additional column for the category to predict (as defined in `classification/config.yaml`).

### Segmentation CSV

| PatientID | img_path | seg_path |
|-----------|----------|----------|
| ID1       | /path/to/image1.nii.gz | /path/to/segmentation1.nii.gz |
| ID2       | /path/to/image2.nii.gz | /path/to/segmentation2.nii.gz |

## Configuration

Each training type has its own `config.yaml` file. Make sure to set the following parameters:

- `results_dir`: Path to save results and checkpoints
- `csv_path`: Path to the CSV file for the respective training type

If using pretrained weights (`pretrained: true`), also set:
- `pretrained_path`: Path to the pretrained weights

Pretrained weights can be downloaded from [SuPreM GitHub repository](https://github.com/MrGiovanni/SuPreM).

## Running the Training

(Add instructions on how to run each type of training)

## Data Pipeline

The data pipeline and operators are defined in the `data/` folder. Refer to `pipeline.py` and `operators.py` for details on data processing.

## Additional Information

(Add any other relevant information, dependencies, or instructions)
