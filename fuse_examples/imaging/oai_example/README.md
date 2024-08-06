# Project Name

Brief description of your project.

## Project Structure

│
├── data/ <br>
│   ├── data_ops.py  <br>
│   ├── oai_ds.py  <br>
│   └── seg_ds.py  <br>
│  <br>
├── downstream/  <br>
│   ├── classification.yaml  <br>
│   ├── classification.py  <br>
│   ├── segmentation.yaml  <br>
│   └── segmentation.py <br>
│ <br>
├── self_supervised/ <br>
│   ├── dino.yaml <br>
│   └── dino.py <br>



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



