


## 3D Medical Imaging Pre-training and Downstream Task Validation

Self-supervision is employed to learn meaningful representations from unlabeled medical imaging data. By pre-training the model on this vast source of information, we equip it with a strong foundation of understanding the underlying data structure. This leads to significantly improved performance and faster convergence when fine-tuning on downstream tasks like classification and segmentation, compared to training from scratch.

This example demonstrates how to pre-train a model on 3D MRI medical imaging using self-supervised learning techniques, specifically DINO, on large datasets. The pre-trained model can then be fine-tuned for downstream tasks such as classification, segmentation, and masked autoencoding. The fine-tuning process is adaptable to various medical imaging tasks, even when working with small datasets.

We use the NIH Osteoarthritis Initiative (OAI) dataset for this example which can be downloaded from [https://nda.nih.gov/oai](https://nda.nih.gov/oai)

## Data Preparation

For each training type (self-supervised, classification, segmentation, masked autoencoding), prepare a CSV file with the following structure:

### Self-Supervised and Classification CSV

| PatientID | path | fold |
|---|---|---|
| ID1       | /path/to/dicom/folder1 | 0 |
| ID2       | /path/to/dicom/folder2 | 1 |

For classification, you can add multiple categorical columns to predict and add them as "cls_targets" in the  `classification_config.yaml`.
For example, you can classify the disease status (Progression, Non-exposed control group) using the V00COHORT label in the OAI dataset.

### Segmentation CSV

| PatientID | img_path | seg_path | fold | max_val (optional - 2D only) |
|---|---|---|---|---|
| ID1       | /path/to/image1.nii.gz | /path/to/segmentation1.nii.gz | train | 974 |  | ID2       | /path/to/image2.nii.gz | /path/to/segmentation2.nii.gz | val | 1.0 |  For 2D segmentation, the `max_val` column is optional and specifies the maximum intensity value in each slice. This allows for normalization without loading the entire 3D volume.

### Masked Autoencoding CSV (2D and 3D)

The format for the masked autoencoding CSV will depend on whether you're working with 2D or 3D data. Refer to the documentation for `mae2d.py` and `mae3d.py` for specific details.

## Configuration

Each training type has its own `config.yaml` file. Make sure to set the following parameters:

- `results_dir`: Path to save results and checkpoints
- `csv_path`: Path to the CSV file for the respective training type
- `experiment`: Name of the experiment and also the name of the results folder
- `train_folds`: List of fold values to use for training (e.g., [0, 1, 2])
- `val_folds`: List of fold values to use for validation (e.g., [3])
- `test_folds`: List of fold values to use for testing (e.g., [4])
- `test_ckpt`: Path to the checkpoint for testing. If set to "null", the model will train using the train and validation sets. If a path is provided, it will perform evaluation on the test set using the given checkpoint.

To load pretrained weights or start from certain checkpoint you need to set only **one** of the following:
- `baseline_weights`: for the 3D case: Path to the backbone pretrained weights from SuPreM (download from [https://github.com/MrGiovanni/SuPreM](https://github.com/MrGiovanni/SuPreM)). For the 2D case: fill True to use imagenet pretrained weights.
- `dino_weights`: Path to the backbone pretrained weights from Dino
- `mae_weights`: Path to the backbone pretrained weights from Masked Auto Encoder task
- `resume_training_from`: Path to training checkpoint
- `test_ckpt`: If set, the test set as defined in `test_folds` will be evaluated using this checkpoint
If none of them are set you will train from scratch

Pretrained weights can be downloaded from [SuPreM GitHub repository](https://github.com/MrGiovanni/SuPreM).

## Training

The training process involves several main steps, including self-supervised pre-training with DINO, fine-tuning for classification, segmentation, and masked autoencoding:

1. Self-supervised pre-training with DINO
2. Fine-tuning for classification
3. Fine-tuning

### 1. Self-Supervised Pre-training

Run DINO pre-training:

```bash
python fuse_examples/imaging/oai_example/self_supervised/dino.py
```

### 2. Classification Fine-tuning

Set dino_weights in classification_config.yaml to the path of the best DINO checkpoint.
Run classification training:

```bash
python fuse_examples/imaging/oai_example/downstream/classification.py
```

### 3. Segmentation Fine-tuning

Set dino_weights in segmentation_config.yaml to the same DINO checkpoint path.
Run segmentation training:
```bash
python fuse_examples/imaging/oai_example/downstream/segmentation.py
```
This process leverages transfer learning, using DINO pre-trained weights to improve performance on downstream tasks.


### Hydra Overrides

Hydra is a powerful framework for configuring experiments. You can override default parameters in your configuration files using command-line arguments.

**Example:**

To override the `batch_size` for DINO pre-training to 16:

```bash
python fuse_examples/imaging/oai_example/self_supervised/dino.py batch_size=16
```

To override multiple parameters:

```bash
python fuse_examples/imaging/oai_example/self_supervised/dino.py batch_size=16 lr=0.001
```


### Monitoring Results

You can track the progress of your training/testing using one of the following methods:

1. TensorBoard:
   To view losses and metrics, run:
   ```
   tensorboard --logdir=<path_to_experiments_directory>
   ```
2. ClearML:
    If ClearML is installed and enabled in your config file (`clearml : True`), you can use it to monitor your results.

    Choose the method that best suits your workflow and preferences.