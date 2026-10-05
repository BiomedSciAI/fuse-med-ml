# 3D Medical Imaging Pre-training and Downstream Task Validation for KneePreM development

**Preprint:** [arXiv:2609.31461](https://arxiv.org/pdf/2609.31461)

This is a self-supervised workflow for learning transferable representations from unlabeled 3D knee MRI. A 3D U-Net is pretrained as a masked autoencoder (MAE): volumetric regions are hidden, the network reconstructs the masked voxels from the remaining context, and the learned encoder is then fine-tuned for classification or segmentation.

This example uses the [Osteoarthritis Initiative (OAI)](https://nda.nih.gov/oai) as the pretraining source. The same workflow can transfer the pretrained encoder to external knee MRI datasets with different sequences, scanners, and labels. The KneePreM evaluation uses fastMRI+ and a clinical arthroscopic partial meniscectomy (APM) cohort for classification, and SKM-TEA and APM for segmentation. Public datasets must be obtained under their own terms; the clinical APM data are not distributed here.

## Workflow

1. **3D masked-autoencoder pretraining:** train the full 3D U-Net to reconstruct masked OAI MRI volumes.
2. **Encoder transfer:** load the MAE checkpoint and transfer the pretrained encoder weights.
3. **Classification fine-tuning:** attach one or more task-specific classification heads and fine-tune the encoder.
4. **Segmentation fine-tuning:** attach a randomly initialized 3D U-Net decoder and fine-tune the complete network.
5. **Deployment:** expose trained classification and segmentation checkpoints through an interactive Model Context Protocol (MCP) workflow.

The workflow described below centers on 3D masked-autoencoder pretraining and transfer of the learned encoder to downstream knee MRI tasks.

## Code Organization

- `data/`: OAI and segmentation data pipelines, including loading, normalization, augmentation, masking, and resizing.
- `self_supervised/mae3d.py`: main KneePreM 3D masked-autoencoder pretraining entry point.
- `self_supervised/mae_config.yaml`: MAE configuration; set it to the 3D values shown below for KneePreM.
- `downstream/classification.py`: 3D encoder fine-tuning for one or more classification targets.
- `downstream/segmentation3d.py`: 3D U-Net fine-tuning for multiclass segmentation.
- `downstream/classification_config.yaml`: classification data, split, optimization, and initialization settings.
- `downstream/segmentation_config.yaml`: segmentation data, split, optimization, and initialization settings.
- `mcp_inference/`: optional single-case and batch inference tools for trained downstream checkpoints.

## Installation

From the repository root:

```bash
pip install -e .[examples]
```

The MCP workflow additionally requires Python 3.10 or newer and the MCP Python SDK:

```bash
pip install "mcp[cli]"
```

## Data Preparation

### Pretraining and Classification

The 3D MAE and classification pipelines read a CSV containing an examination identifier, the path to a DICOM series directory, and a split or fold value:

| accession_number | path | fold |
| --- | --- | --- |
| ID1 | /path/to/dicom/series_1 | 0 |
| ID2 | /path/to/dicom/series_2 | 1 |

For classification, add one column for every target listed in `cls_targets`:

| accession_number | path | fold | target_1 | target_2 |
| --- | --- | --- | --- | --- |
| ID1 | /path/to/dicom/series_1 | train | 0 | 1 |
| ID2 | /path/to/dicom/series_2 | val | 1 | 0 |

The values in `train_folds`, `val_folds`, and `test_folds` must match the values in the CSV. Keep examinations from the same participant in only one split.

### Segmentation

The 3D segmentation pipeline reads NIfTI image and label volumes:

| idx | img_path | seg_path | fold |
| --- | --- | --- | --- |
| ID1 | /path/to/image_1.nii.gz | /path/to/mask_1.nii.gz | train |
| ID2 | /path/to/image_2.nii.gz | /path/to/mask_2.nii.gz | val |
| ID3 | /path/to/image_3.nii.gz | /path/to/mask_3.nii.gz | test |

Set `num_classes` to the number of segmentation labels including background. Image and mask pairs are resized together, with label-preserving interpolation for masks.

## Configuration

The training scripts use Hydra. You can edit the YAML files or override values on the command line. The most commonly changed fields are:

- `csv_path`: input manifest.
- `results_dir`: output directory for logs and checkpoints.
- `experiment`: experiment name and results subdirectory.
- `train_folds`, `val_folds`, and `test_folds`: dataset split values.
- `resize_to`: standardized input volume size.
- `cuda_devices`, `batch_size`, `n_workers`, `learning_rate`, `weight_decay`, and `n_epochs`: runtime and optimization settings.
- `test_ckpt`: downstream checkpoint to evaluate instead of training.

## 1. Pretrain KneePreM with 3D Masked Autoencoding

The KneePreM setup uses a 3D U-Net, inputs resized to `40 x 224 x 224`, and cuboid masking. The following command supplies the main 3D settings explicitly:

```bash
python fuse_examples/imaging/oai_example/self_supervised/mae3d.py \
  backbone=unet3d \
  'resize_to=[40,224,224]' \
  mae_cfg.mask_percentage=0.64 \
  'mae_cfg.cuboid_size=[4,4,4]' \
  batch_size=4 \
  learning_rate=0.00005 \
  weight_decay=0.001 \
  n_epochs=200 \
  pretrained=false \
  csv_path=/path/to/oai_pretraining.csv \
  results_dir=/path/to/results \
  experiment=kneeprem_oai_mae3d
```

Five percent of the OAI pretraining source can be reserved for validation by assigning it a separate fold and setting `val_folds` accordingly. Pretraining uses images only; no manual labels are required for the reconstruction objective.

## 2. Fine-tune for Classification

Set the dataset, targets, and splits in `downstream/classification_config.yaml`. To initialize the 3D encoder with KneePreM, set `mae_weights` to the MAE checkpoint and leave the other initialization fields unset:

```yaml
backbone: unet3d
resize_to: [40, 224, 224]
cls_targets: [target_1, target_2]
mae_weights: /path/to/kneeprem_mae_checkpoint.ckpt
suprem_weights: null
resume_training_from: null
test_ckpt: null
```

Run training:

```bash
python fuse_examples/imaging/oai_example/downstream/classification.py
```

The same encoder can support one or multiple classification heads by changing `cls_targets` and adding the corresponding columns to the CSV.

## 3. Fine-tune for 3D Segmentation

Set the dataset, label count, and splits in `downstream/segmentation_config.yaml`. To initialize with KneePreM:

```yaml
backbone: unet3d
resize_to: [40, 224, 224]
mae_weights: /path/to/kneeprem_mae_checkpoint.ckpt
baseline_weights: null
resume_training_from: null
test_ckpt: null
```

Run training:

```bash
python fuse_examples/imaging/oai_example/downstream/segmentation3d.py
```

The transferred encoder is combined with a segmentation decoder and the complete 3D network is fine-tuned on the labeled volumes.

## Initialization Comparisons

Use only one initialization or checkpoint field at a time.

| Initialization | Classification config | Segmentation config |
| --- | --- | --- |
| KneePreM | `mae_weights` | `mae_weights` |
| SuPreM | `suprem_weights` | `baseline_weights` |
| Random | leave all weight fields `null` | leave all weight fields `null` |

Compatible SuPreM weights are available from the [SuPreM repository](https://github.com/MrGiovanni/SuPreM).

## Monitoring Training

TensorBoard logs can be viewed with:

```bash
tensorboard --logdir=/path/to/results
```

ClearML can also be enabled with `clearml=true` and a configured `clearml_project_name`.

## MCP Inference for Segmentation and Classification

The current inference workflow supports downstream segmentation, downstream classification, or both in one run. It can be used as:

- a persistent interactive CLI session backed by a background MCP HTTP server
- a protocol-level MCP server using the official MCP Python SDK

In this example, the workflow is organized as a set of callable tools with clear inputs and outputs:

- preprocessing
- segmentation
- classification
- QC visualization
- result logging

The workflow uses the official MCP Python SDK and serves tool-based inference over Streamable HTTP.

### Code Organization

The inference functionality has been modularized into four main components for better maintainability:

- `inference_utils.py`: Contains utilities, data models, configuration handling, and helper functions
- `inference_tools.py`: Implements the core inference tools (preprocessing, segmentation, classification, visualization) and the MCP inference engine
- `inference_cli.py`: Main entry point, MCP server setup, and interactive CLI interface
- `agent_example.py`: Runnable observe-plan-act client that discovers and invokes the MCP tools

### Required Packages

Python 3.10 or newer is required for this MCP workflow because `mcp[cli]` does not support Python 3.9.

From the repository root, install the project and example dependencies with:

```bash
pip install -e .[examples]
pip install "mcp[cli]"
```

For this workflow specifically:

- `pip install -e .[examples]` makes the local `fuse-med-ml` package importable and installs the example runtime dependencies used here, including `torch`, `numpy`, `pandas`, `matplotlib`, `nibabel`, and `monai`
- `pip install "mcp[cli]"` adds the MCP server/client SDK used by the interactive CLI and Streamable HTTP server
- `pydicom` is only needed when your input is a DICOM folder instead of a `.nii` / `.nii.gz` volume, and it is already included in `.[examples]`

Before running inference, set compatible downstream checkpoint paths and class labels in `mcp_inference/inference_config.yaml`; model checkpoints are not distributed with the repository.

Launch the interactive workflow:

```bash
python fuse_examples/imaging/oai_example/mcp_inference/inference_cli.py
```

By default, this starts a background MCP server and then opens the interactive terminal workflow against that same server. While the session is open, other MCP clients can connect to the same host, port, and path.

The default output root is `fuse_examples/imaging/oai_example/outputs/mcp_inference/`. Each run creates a `session_<timestamp>/` folder, and cases are written to neutral subfolders such as `case_0001/`, `case_0002/`, and so on. The original input-derived `case_id` is still preserved inside the CSV/JSON metadata for traceability. A typical session contains per-case NIfTI/PNG/JSON outputs together with a session-level `inference_log.csv`, as summarized below.

You can optionally point to a different config or device:

```bash
python fuse_examples/imaging/oai_example/mcp_inference/inference_cli.py \
  --inference-config fuse_examples/imaging/oai_example/mcp_inference/inference_config.yaml \
  --device auto
```

To run only the Streamable HTTP MCP server:

```bash
python fuse_examples/imaging/oai_example/mcp_inference/inference_cli.py \
  --serve-mcp \
  --host 127.0.0.1 \
  --port 8000
```

Under the hood, the workflow exposes these MCP tools over Streamable HTTP:

- `get_inference_settings`
- `update_inference_settings`
- `reset_inference_settings`
- `process_case`
- `process_batch`

Typical MCP inputs:

- `path` for `process_case`
- `batch_path` for `process_batch`
- `task`: one of `segmentation`, `classification`, or `all`
- `qc_visualization`: optional boolean
- `output_dir`: optional output root
- `log_to_csv`: optional boolean

At startup the session shows the current defaults and offers:

1. Process input
2. Change settings
3. Reset to defaults
4. View current settings
5. Exit

The interactive terminal view looks like this:

```text
Background MCP server ready at http://127.0.0.1:8000/mcp

Interactive inference workflow
Defaults: mode=single, task=all, input=nifti, qc=on, logging=on
1. Process input
2. Change settings
3. Reset to defaults
4. View current settings
5. Exit
Choose an option [1]:
```

Input handling:

- single mode accepts one `.nii` / `.nii.gz` volume path, and also supports a DICOM folder if needed
- batch mode accepts either a folder of cases or a `.csv`, `.tsv`, `.txt`, or `.jsonl` manifest with a `path`, `input_path`, or `img_path` column
- batch processing continues if one case fails and records the failure in the run log

Supported tasks:

- segmentation only
- classification only
- all tasks in sequence

Outputs:

- `segmentation_mask.nii.gz` when segmentation is enabled
- `segmentation_qc.png` when QC is enabled
- `classification.json` when classification is enabled
- `inference_metadata.json` with input and preprocessing metadata
- `inference_log.csv` with per-case status and output paths

### Agent Example

`agent_example.py` demonstrates a complete, bounded agent loop:

1. **Observe** the live MCP tool catalogue and inference settings.
2. **Plan** a `process_case` action for one input or `process_batch` for several inputs.
3. **Act** by configuring the server and invoking only the selected MCP tool.
4. **Report** classifications, segmentation label counts, output paths, and the action trace in `agent_report.json`.

The example is deterministic and does not require an API key or send image data to an external LLM. By default it starts a temporary local MCP server and stops it when the run finishes. Model checkpoints are not distributed with the repository, so provide paths to compatible downstream checkpoints.

Run three NIfTI cases from a directory through both downstream tasks:

```bash
python fuse_examples/imaging/oai_example/mcp_inference/agent_example.py \
  --sample-dir /path/to/nifti_cases \
  --limit 3 \
  --task all \
  --classification-weights /path/to/classification.ckpt \
  --segmentation-weights /path/to/segmentation.ckpt \
  --output-dir /path/to/agent_outputs \
  --device cuda:0
```

Pass `--input` more than once to choose exact cases:

```bash
python fuse_examples/imaging/oai_example/mcp_inference/agent_example.py \
  --input /path/to/case_a.nii.gz \
  --input /path/to/case_b.nii.gz \
  --task segmentation \
  --segmentation-weights /path/to/segmentation.ckpt
```

To use a separately managed server instead, add:

```bash
--server-url http://127.0.0.1:8000/mcp
```

Clinical datasets are not distributed with this repository.
