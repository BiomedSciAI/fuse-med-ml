import argparse
import asyncio
import ast
import contextlib
import csv
import io
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import pandas as pd

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    FastMCP = None

try:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
except ImportError:
    ClientSession = None
    streamable_http_client = None


BASE_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

OAI_EXAMPLE_DIR = os.path.dirname(BASE_DIR)
DEFAULT_INFERENCE_CONFIG_PATH = os.path.join(BASE_DIR, "inference_config.yaml")
DEFAULT_OUTPUT_DIR = os.path.join(OAI_EXAMPLE_DIR, "outputs", "mcp_inference")
DEFAULT_CLASSIFICATION_WEIGHTS = os.path.join(BASE_DIR, "weights", "classification_model.ckpt")
DEFAULT_SEGMENTATION_WEIGHTS = os.path.join(BASE_DIR, "weights", "segmentation_model.ckpt")
LOG_FIELDNAMES = [
    "timestamp",
    "case_id",
    "case_directory_name",
    "input_path",
    "task",
    "mode",
    "preprocessing_status",
    "model_name",
    "weights_path",
    "classification_model_name",
    "classification_weights_path",
    "segmentation_model_name",
    "segmentation_weights_path",
    "predicted_label",
    "predicted_probability",
    "segmentation_mask_path",
    "qc_image_path",
    "classification_json_path",
    "output_directory",
    "status",
    "error_message",
]


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _unwrap_mcp_payload(payload: Any) -> Any:
    if isinstance(payload, dict) and set(payload.keys()) == {"result"}:
        return payload["result"]
    return payload


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device(requested_device)


def _parse_config_value(value: str) -> Any:
    normalized = value.strip()
    lowered = normalized.lower()
    if lowered == "null":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    if normalized.startswith("[") and normalized.endswith("]"):
        inner = normalized[1:-1].strip()
        if not inner:
            return []
        parts = [part.strip() for part in inner.split(",")]
        return [_parse_config_value(part) for part in parts]

    try:
        return ast.literal_eval(normalized)
    except (ValueError, SyntaxError):
        return normalized.strip("'\"")


def _load_config(config_path: str) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    if not os.path.exists(config_path):
        return config

    with open(config_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line or raw_line[:1].isspace():
                continue
            if ":" not in line:
                continue

            key, raw_value = line.split(":", 1)
            key = key.strip()
            value = raw_value.strip()
            if not value:
                continue
            config[key] = _parse_config_value(value)

    return config


def _state_dict_candidates(checkpoint: Any) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model_state_dict"]:
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                candidates.append(dict(candidate))

        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            candidates.append(dict(checkpoint))

    prefixes = ["_model.", "model.", "module.", "_orig_mod.", "_forward_module."]
    expanded: List[Dict[str, Any]] = []
    for candidate in candidates:
        expanded.append(candidate)
        for prefix in prefixes:
            stripped = {
                key[len(prefix) :]: value
                for key, value in candidate.items()
                if key.startswith(prefix)
            }
            if stripped:
                expanded.append(stripped)
    return expanded


def _choose_state_dict(
    raw_state_dict: Dict[str, Any], model_keys: Sequence[str]
) -> Dict[str, Any]:
    model_key_set = set(model_keys)
    candidates = [dict(raw_state_dict)]

    prefixes = ["_model.", "model.", "module.", "_orig_mod.", "_forward_module."]
    for prefix in prefixes:
        stripped = {
            key[len(prefix) :]: value
            for key, value in raw_state_dict.items()
            if key.startswith(prefix)
        }
        if stripped:
            candidates.append(stripped)

    def overlap(candidate: Dict[str, Any]) -> int:
        return len(set(candidate.keys()) & model_key_set)

    return max(candidates, key=overlap)


def _extract_checkpoint_state_dict(
    checkpoint: Any, model_keys: Sequence[str]
) -> Dict[str, Any]:
    for candidate in _state_dict_candidates(checkpoint):
        chosen = _choose_state_dict(candidate, model_keys)
        if chosen:
            return chosen

    raise ValueError(
        "Unsupported checkpoint format. Expected a Lightning checkpoint or a raw state dict."
    )


def _infer_head_output_dims(checkpoint: Any, cls_targets: Sequence[str]) -> Dict[str, int]:
    named_pattern = re.compile(
        r"^heads\.head_([^\.]+)\.conv_classifier_3d\.classifier\.(\d+)\.weight$"
    )
    indexed_head_pattern = re.compile(r"^heads\.(\d+)\.")
    indexed_pattern = re.compile(
        r"^heads\.(\d+)\.conv_classifier_3d\.classifier\.(\d+)\.weight$"
    )
    errors: List[str] = []

    for candidate in _state_dict_candidates(checkpoint):
        named_dims: Dict[str, List[Tuple[int, int]]] = {}
        indexed_dims: Dict[int, List[Tuple[int, int]]] = {}
        indexed_head_indices = set()

        for key, value in candidate.items():
            indexed_head_match = indexed_head_pattern.match(key)
            if indexed_head_match:
                indexed_head_indices.add(int(indexed_head_match.group(1)))

            if not torch.is_tensor(value) or value.ndim != 5:
                continue

            named_match = named_pattern.match(key)
            if named_match:
                target_name = named_match.group(1)
                if target_name in cls_targets:
                    named_dims.setdefault(target_name, []).append(
                        (int(named_match.group(2)), int(value.shape[0]))
                    )
                continue

            indexed_match = indexed_pattern.match(key)
            if indexed_match:
                indexed_dims.setdefault(int(indexed_match.group(1)), []).append(
                    (int(indexed_match.group(2)), int(value.shape[0]))
                )

        if all(target in named_dims for target in cls_targets):
            return {target: max(named_dims[target])[1] for target in cls_targets}

        if indexed_head_indices:
            if not indexed_dims:
                errors.append(
                    "Checkpoint contains indexed classification heads but no "
                    "classifier weight layers were found."
                )
                continue

            if len(indexed_head_indices) < len(cls_targets):
                errors.append(
                    "Checkpoint contains "
                    f"{len(indexed_head_indices)} indexed classification head(s), "
                    f"but {len(cls_targets)} classification target(s) were requested: "
                    f"{list(cls_targets)}."
                )
                continue

            missing_indices = [
                index for index in range(len(cls_targets)) if index not in indexed_dims
            ]
            if missing_indices:
                errors.append(
                    "Checkpoint is missing classifier weight layers for indexed "
                    f"head(s) {missing_indices}. Available indexed head(s): "
                    f"{sorted(indexed_dims)}."
                )
                continue

            return {
                target: max(indexed_dims[index])[1]
                for index, target in enumerate(cls_targets)
            }

        if named_dims:
            available_named_targets = sorted(named_dims)
            missing_targets = [
                target for target in cls_targets if target not in available_named_targets
            ]
            errors.append(
                "Checkpoint contains named classification heads for "
                f"{available_named_targets}, but missing requested target(s): "
                f"{missing_targets}."
            )

    if errors:
        raise RuntimeError(errors[0])

    raise RuntimeError(
        "Failed to infer classification head sizes from checkpoint. "
        "Provide a compatible downstream classification checkpoint."
    )


def _resolve_path(path: Optional[str], base_dir: str) -> Optional[str]:
    if path is None:
        return None
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def _input_case_id(input_path: str) -> str:
    normalized = os.path.normpath(input_path)
    if os.path.isdir(normalized):
        return os.path.basename(normalized)
    filename = os.path.basename(normalized)
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    return os.path.splitext(filename)[0]


def _case_directory_name(case_index: int) -> str:
    if case_index < 1:
        raise ValueError("case_index must be a positive integer")
    return f"case_{case_index:04d}"


def _parse_bool(text: str, default: bool) -> bool:
    value = text.strip().lower()
    if value == "":
        return default
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected yes/no style input, got: {text}")


def _prompt(prompt_text: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default not in (None, "") else ""
    value = input(f"{prompt_text}{suffix}: ").strip()
    if value == "" and default is not None:
        return default
    return value


def _load_batch_inputs(batch_path: str) -> List[str]:
    batch_path = os.path.expanduser(batch_path)
    manifest_dir = os.path.dirname(os.path.abspath(batch_path))

    def _resolve_manifest_path(path: str) -> str:
        path = os.path.expanduser(str(path))
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(manifest_dir, path))

    if os.path.isdir(batch_path):
        entries = []
        for name in sorted(os.listdir(batch_path)):
            candidate = os.path.join(batch_path, name)
            if os.path.isdir(candidate) or candidate.endswith((".nii", ".nii.gz")):
                entries.append(candidate)
        if not entries:
            raise ValueError(f"No case folders or NIfTI files found in {batch_path}")
        return entries

    if not os.path.isfile(batch_path):
        raise FileNotFoundError(f"Batch input not found: {batch_path}")

    lower_path = batch_path.lower()
    if lower_path.endswith(".jsonl"):
        paths = []
        with open(batch_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                path = payload.get("path")
                if path:
                    paths.append(_resolve_manifest_path(path))
        if not paths:
            raise ValueError(f"No 'path' values found in {batch_path}")
        return paths

    if lower_path.endswith((".csv", ".tsv", ".txt")):
        sep = "\t" if lower_path.endswith(".tsv") else None
        df = pd.read_csv(batch_path, sep=sep)
        for column in ["path", "input_path", "img_path"]:
            if column in df.columns:
                return [_resolve_manifest_path(path) for path in df[column].dropna().astype(str)]
        if len(df.columns) == 0:
            raise ValueError(f"No columns found in {batch_path}")
        return [_resolve_manifest_path(path) for path in df.iloc[:, 0].dropna().astype(str)]

    raise ValueError(
        "Batch input must be a directory, .csv, .tsv, .txt, or .jsonl manifest."
    )


@dataclass
class SessionSettings:
    input_mode: str
    task: str
    input_format: str
    classification_weights_path: str
    segmentation_weights_path: str
    classification_model_name: str
    segmentation_model_name: str
    classification_cls_targets: List[str]
    classification_class_labels: Dict[str, List[Any]]
    segmentation_num_classes: int
    preprocessing_resize_to: Tuple[int, int, int]
    qc_visualization: bool
    csv_logging: bool
    output_dir: str
    device: str
    inference_config_path: str


@dataclass
class PreparedCase:
    case_id: str
    input_path: str
    source_type: str
    original_shape: Tuple[int, ...]
    resized_shape: Tuple[int, ...]
    tensor: torch.Tensor
    resized_image: np.ndarray
    affine: np.ndarray


class PreprocessingTool:
    """Load, normalize, and resize MRI volumes for downstream inference."""

    def __init__(self, resize_to: Sequence[int]):
        self.resize_to = tuple(int(value) for value in resize_to)

    def prepare(self, input_path: str, case_id: Optional[str] = None) -> PreparedCase:
        from scipy.ndimage import zoom

        resolved_path = os.path.expanduser(input_path)
        if os.path.isdir(resolved_path):
            try:
                import pydicom
            except ImportError as exc:
                raise ImportError(
                    "pydicom is required only for DICOM-folder inputs. "
                    "Install it or provide a NIfTI file instead."
                ) from exc
            dicom_files = sorted(
                file_name
                for file_name in os.listdir(resolved_path)
                if os.path.isfile(os.path.join(resolved_path, file_name))
            )
            if not dicom_files:
                raise ValueError(f"No files found in DICOM folder: {resolved_path}")

            slices = []
            for file_name in dicom_files:
                dicom = pydicom.dcmread(os.path.join(resolved_path, file_name))
                slices.append(np.asarray(dicom.pixel_array))
            image = np.stack(slices, axis=0)
            source_type = "dicom_dir"
            affine = np.eye(4, dtype=np.float32)
        elif os.path.isfile(resolved_path) and resolved_path.endswith((".nii", ".nii.gz")):
            nifti = nib.load(resolved_path)
            image = np.asarray(nifti.get_fdata())
            source_type = "nifti_file"
            affine = nifti.affine
        else:
            raise FileNotFoundError(
                "Input must be an existing DICOM folder or a .nii/.nii.gz file, "
                f"got: {input_path}"
            )

        image = image.astype(np.float32)
        original_shape = tuple(int(dim) for dim in image.shape)
        image = np.clip(image, *(np.percentile(image, [0, 95])))
        image -= image.min()
        max_value = image.max()
        if max_value == 0:
            raise RuntimeError(f"Normalization failed for input: {input_path}")
        image /= max_value

        resize_factors = [
            target_dim / current_dim
            for target_dim, current_dim in zip(self.resize_to, image.shape)
        ]
        image = zoom(image, resize_factors, order=1)
        image = np.ascontiguousarray(image).astype(np.float32)
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        return PreparedCase(
            case_id=case_id or _input_case_id(resolved_path),
            input_path=resolved_path,
            source_type=source_type,
            original_shape=original_shape,
            resized_shape=tuple(int(dim) for dim in image.shape),
            tensor=tensor,
            resized_image=image,
            affine=affine,
        )


class ClassificationTool:
    """Wrap the existing downstream classification inference runner."""

    def __init__(
        self,
        checkpoint_path: str,
        model_name: str,
        cls_targets: Sequence[str],
        class_labels: Optional[Dict[str, Sequence[Any]]],
        device: str,
    ) -> None:
        from fuse.dl.models import ModelMultiHead
        from fuse.dl.models.backbones.backbone_unet3d import UNet3D
        from fuse.dl.models.heads.heads_3D import Head3D

        self.model_name = model_name
        self.weights_path = os.path.expanduser(checkpoint_path)
        self.cls_targets = list(cls_targets)
        self.device = _resolve_device(device)

        checkpoint = torch.load(self.weights_path, map_location="cpu")
        inferred_output_dims = _infer_head_output_dims(checkpoint, self.cls_targets)

        self.classes_by_target: Dict[str, List[Any]] = {}
        class_labels = class_labels or {}
        for target in self.cls_targets:
            labels = list(class_labels.get(target, []))
            if labels and len(labels) != inferred_output_dims[target]:
                raise ValueError(
                    f"classification_class_labels[{target}] has {len(labels)} entries, "
                    f"but the checkpoint expects {inferred_output_dims[target]} outputs."
                )
            if not labels:
                labels = [str(index) for index in range(inferred_output_dims[target])]
            self.classes_by_target[target] = [_jsonable_value(label) for label in labels]

        backbone = UNet3D(for_cls=True)
        conv_inputs = [("model.backbone_features", 512)]
        heads = [
            Head3D(
                head_name=f"head_{target}",
                mode="classification",
                conv_inputs=conv_inputs,
                num_outputs=len(self.classes_by_target[target]),
            )
            for target in self.cls_targets
        ]
        self.model = ModelMultiHead(conv_inputs=(("img", 1),), backbone=backbone, heads=heads)

        state_dict = _extract_checkpoint_state_dict(
            checkpoint=checkpoint,
            model_keys=self.model.state_dict().keys(),
        )
        matching_keys = set(state_dict.keys()) & set(self.model.state_dict().keys())
        if not matching_keys:
            raise RuntimeError(
                "No checkpoint weights matched the classification model. "
                "Make sure the checkpoint is from the downstream classification task."
            )

        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        missing_backbone_or_heads = [
            key
            for key in missing_keys
            if key.startswith("backbone.") or key.startswith("heads.")
        ]
        if missing_backbone_or_heads:
            raise RuntimeError(
                "Checkpoint is missing classification model weights. "
                f"First missing keys: {missing_backbone_or_heads[:5]}"
            )
        if unexpected_keys:
            print("Ignoring unexpected checkpoint keys:", ", ".join(unexpected_keys[:5]))

        self.model.to(self.device)
        self.model.eval()

    def predict(self, prepared_case: PreparedCase) -> Dict[str, Any]:
        from fuse.utils import NDict

        batch_dict = NDict({"img": prepared_case.tensor.to(self.device)})
        with torch.inference_mode():
            batch_dict = self.model(batch_dict)

        result: Dict[str, Any] = {
            "case_id": prepared_case.case_id,
            "input_path": prepared_case.input_path,
            "targets": {},
        }
        for target in self.cls_targets:
            probabilities = (
                batch_dict[f"model.output.head_{target}"][0].detach().cpu().tolist()
            )
            logits = batch_dict[f"model.logits.head_{target}"][0].detach().cpu().tolist()
            predicted_index = int(np.argmax(probabilities))
            class_labels = self.classes_by_target[target]
            result["targets"][target] = {
                "predicted_index": predicted_index,
                "predicted_label": _jsonable_value(class_labels[predicted_index]),
                "class_labels": [_jsonable_value(value) for value in class_labels],
                "probabilities": [float(value) for value in probabilities],
                "logits": [float(value) for value in logits],
            }

        primary_target = self.cls_targets[0]
        result["primary_target"] = primary_target
        result["predicted_label"] = result["targets"][primary_target]["predicted_label"]
        result["predicted_probability"] = max(
            result["targets"][primary_target]["probabilities"]
        )
        return result


class SegmentationTool:
    """Run 3D downstream segmentation inference using the existing model definition."""

    def __init__(
        self,
        checkpoint_path: str,
        model_name: str,
        num_classes: int,
        device: str,
    ) -> None:
        from fuse.dl.models import ModelMultiHead
        from fuse.dl.models.backbones.backbone_unet3d import UNet3D
        from fuse.dl.models.heads.head_dense_segmentation import HeadDenseSegmentation
        import torch.nn as nn

        self.weights_path = os.path.expanduser(checkpoint_path)
        self.device = _resolve_device(device)
        self.num_classes = int(num_classes)
        self.model_name = model_name

        backbone = UNet3D(for_cls=False)
        heads = [
            HeadDenseSegmentation(
                head_name="head_seg",
                conv_inputs=[("model.backbone_features", 512)],
                shared_classifier_head=nn.Conv3d(64, self.num_classes, kernel_size=1),
            )
        ]
        self.model = ModelMultiHead(conv_inputs=(("img", 1),), backbone=backbone, heads=heads)
        self._load_checkpoint()
        self.model.to(self.device)
        self.model.eval()

    def _load_checkpoint(self) -> None:
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Segmentation checkpoint not found: {self.weights_path}")

        checkpoint = torch.load(self.weights_path, map_location="cpu")
        state_dict = _extract_checkpoint_state_dict(
            checkpoint=checkpoint,
            model_keys=self.model.state_dict().keys(),
        )
        missing_keys, _ = self.model.load_state_dict(state_dict, strict=False)
        missing_model_keys = [
            key
            for key in missing_keys
            if key.startswith("backbone.") or key.startswith("heads.")
        ]
        if missing_model_keys:
            raise RuntimeError(
                "Checkpoint is missing segmentation model weights. "
                f"First missing keys: {missing_model_keys[:5]}"
            )

    def predict(self, prepared_case: PreparedCase) -> np.ndarray:
        from fuse.utils import NDict

        batch_dict = NDict({"img": prepared_case.tensor.to(self.device)})
        with torch.inference_mode():
            batch_dict = self.model(batch_dict)

        logits = batch_dict["model.logits.head_seg"]
        if logits.ndim == 5:
            logits = logits[0]
        mask = logits.argmax(dim=0).detach().cpu().numpy().astype(np.uint8)
        return mask


class VisualizationTool:
    """Create lightweight QC overlays for segmentation output."""

    def save_segmentation_qc(
        self, prepared_case: PreparedCase, mask: np.ndarray, output_path: str
    ) -> str:
        center_idx = prepared_case.resized_image.shape[0] // 2
        image_slice = prepared_case.resized_image[center_idx]
        mask_slice = mask[center_idx]
        foreground_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
        max_label = max(int(mask.max()), 1)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image_slice, cmap="gray")
        axes[0].set_title("MRI")
        axes[0].axis("off")

        axes[1].imshow(image_slice, cmap="gray")
        axes[1].imshow(
            foreground_overlay,
            cmap="tab10",
            alpha=0.65,
            interpolation="nearest",
            vmin=1,
            vmax=max_label,
        )
        axes[1].set_title("QC Overlay")
        axes[1].axis("off")

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path


class ResultLoggerTool:
    """Append per-case run metadata to CSV."""

    def append(self, log_path: str, row: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_exists = os.path.exists(log_path)
        with open(log_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LOG_FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in LOG_FIELDNAMES})


class InteractiveInferenceWorkflow:
    """Session controller for tool-based downstream inference orchestration."""

    def __init__(self, settings: SessionSettings) -> None:
        self.settings = settings
        self.preprocessing_tool = PreprocessingTool(
            resize_to=self.settings.preprocessing_resize_to
        )
        self.visualization_tool = VisualizationTool()
        self.logger_tool = ResultLoggerTool()
        self._classification_tool: Optional[ClassificationTool] = None
        self._segmentation_tool: Optional[SegmentationTool] = None
        self._classification_cache_key: Optional[Tuple[str, str, str]] = None
        self._segmentation_cache_key: Optional[Tuple[str, str, str]] = None

    def get_settings_dict(self) -> Dict[str, Any]:
        return asdict(self.settings)

    def _invalidate_model_caches(self) -> None:
        self._classification_tool = None
        self._segmentation_tool = None
        self._classification_cache_key = None
        self._segmentation_cache_key = None

    def update_settings(
        self,
        *,
        input_mode: Optional[str] = None,
        task: Optional[str] = None,
        input_format: Optional[str] = None,
        classification_weights_path: Optional[str] = None,
        segmentation_weights_path: Optional[str] = None,
        qc_visualization: Optional[bool] = None,
        csv_logging: Optional[bool] = None,
        output_dir: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        if input_mode is not None:
            normalized_input_mode = input_mode.lower()
            if normalized_input_mode not in {"single", "batch"}:
                raise ValueError("input_mode must be one of: single, batch")
            self.settings.input_mode = normalized_input_mode

        if task is not None:
            normalized_task = task.lower()
            if normalized_task not in {"segmentation", "classification", "all"}:
                raise ValueError("task must be one of: segmentation, classification, all")
            self.settings.task = normalized_task

        if input_format is not None:
            self.settings.input_format = input_format.lower()

        if classification_weights_path is not None:
            self.settings.classification_weights_path = classification_weights_path

        if segmentation_weights_path is not None:
            self.settings.segmentation_weights_path = segmentation_weights_path

        if qc_visualization is not None:
            self.settings.qc_visualization = qc_visualization

        if csv_logging is not None:
            self.settings.csv_logging = csv_logging

        if output_dir is not None:
            self.settings.output_dir = output_dir

        if device is not None:
            self.settings.device = device

        self._invalidate_model_caches()
        return self.get_settings_dict()

    def run(self) -> None:
        while True:
            self._print_menu()
            choice = _prompt("Choose an option", "1")
            if choice == "1":
                self.process_inputs()
            elif choice == "2":
                self.change_settings()
            elif choice == "3":
                self.reset_defaults()
            elif choice == "4":
                self.view_settings()
            elif choice == "5":
                print("Exiting interactive inference session.")
                return
            else:
                print("Please select 1, 2, 3, 4, or 5.")

    def _print_menu(self) -> None:
        print("\nInteractive inference workflow")
        self.view_settings(compact=True)
        print("1. Process input")
        print("2. Change settings")
        print("3. Reset to defaults")
        print("4. View current settings")
        print("5. Exit")

    def view_settings(self, compact: bool = False) -> None:
        settings = asdict(self.settings)
        if compact:
            print(
                "Defaults: "
                f"mode={settings['input_mode']}, "
                f"task={settings['task']}, "
                f"input={settings['input_format']}, "
                f"qc={'on' if settings['qc_visualization'] else 'off'}, "
                f"logging={'on' if settings['csv_logging'] else 'off'}"
            )
            return

        print("\nCurrent settings")
        for key, value in settings.items():
            print(f"- {key}: {value}")

    def reset_defaults(self) -> Dict[str, Any]:
        self.settings = build_default_settings(
            inference_config_path=self.settings.inference_config_path,
            device=self.settings.device,
        )
        self._invalidate_model_caches()
        print("Defaults restored.")
        return self.get_settings_dict()

    def change_settings(self) -> None:
        print("\nUpdate settings. Press Enter to keep the current value.")
        input_mode = _prompt(
            "Input mode (single/batch)", self.settings.input_mode
        ).lower()
        task = _prompt(
            "Task (segmentation/classification/all)", self.settings.task
        ).lower()
        input_format = _prompt(
            "Input format hint (nifti/dicom/mixed)", self.settings.input_format
        ).lower()
        classification_weights_path = _prompt(
            "Classification weights path",
            self.settings.classification_weights_path,
        )
        segmentation_weights_path = _prompt(
            "Segmentation weights path",
            self.settings.segmentation_weights_path,
        )
        qc_visualization = _parse_bool(
            _prompt(
                "QC visualization (on/off)",
                "on" if self.settings.qc_visualization else "off",
            ),
            self.settings.qc_visualization,
        )
        csv_logging = _parse_bool(
            _prompt(
                "CSV logging (on/off)",
                "on" if self.settings.csv_logging else "off",
            ),
            self.settings.csv_logging,
        )
        output_dir = _prompt("Output directory", self.settings.output_dir)
        device = _prompt("Device", self.settings.device)

        self.update_settings(
            input_mode=input_mode,
            task=task,
            input_format=input_format,
            classification_weights_path=classification_weights_path,
            segmentation_weights_path=segmentation_weights_path,
            qc_visualization=qc_visualization,
            csv_logging=csv_logging,
            output_dir=output_dir,
            device=device,
        )
        print("Settings updated.")

    def process_inputs(self) -> None:
        if self.settings.input_mode == "single":
            input_path = _prompt("Case input path")
            input_paths = [input_path]
        elif self.settings.input_mode == "batch":
            batch_path = _prompt("Batch folder or manifest path")
            input_paths = _load_batch_inputs(batch_path)
            print(f"Loaded {len(input_paths)} case(s) for batch processing.")
        else:
            raise ValueError(f"Unsupported input mode: {self.settings.input_mode}")

        run_dir, log_path = self._create_run_context(self.settings.output_dir)

        success_count = 0
        failure_count = 0
        for case_index, input_path in enumerate(input_paths, start=1):
            case_dir_name = _case_directory_name(case_index)
            try:
                self._process_single_case(
                    input_path=input_path,
                    run_dir=run_dir,
                    log_path=log_path,
                    task=self.settings.task,
                    qc_visualization=self.settings.qc_visualization,
                    log_to_csv=self.settings.csv_logging,
                    case_directory_name=case_dir_name,
                )
                success_count += 1
            except Exception as exc:
                failure_count += 1
                case_id = _input_case_id(input_path)
                case_dir = os.path.join(run_dir, case_dir_name)
                os.makedirs(case_dir, exist_ok=True)
                row = self._base_log_row(
                    case_id=case_id,
                    case_directory_name=case_dir_name,
                    input_path=input_path,
                    output_directory=case_dir,
                    task=self.settings.task,
                    status="failed",
                    error_message=str(exc),
                )
                if self.settings.csv_logging:
                    self.logger_tool.append(log_path=log_path, row=row)
                print(f"[FAILED] {case_id}: {exc}")

        print(
            f"Run finished. successes={success_count}, failures={failure_count}, "
            f"outputs={run_dir}"
        )

    def _create_run_context(self, output_dir: str) -> Tuple[str, str]:
        run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(output_dir, f"session_{run_timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir, os.path.join(run_dir, "inference_log.csv")

    def _base_log_row(
        self,
        case_id: str,
        case_directory_name: str,
        input_path: str,
        output_directory: str,
        task: str,
        status: str,
        error_message: str = "",
    ) -> Dict[str, Any]:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "case_id": case_id,
            "case_directory_name": case_directory_name,
            "input_path": input_path,
            "task": task,
            "mode": self.settings.input_mode,
            "preprocessing_status": "pending",
            "model_name": "",
            "weights_path": "",
            "classification_model_name": "",
            "classification_weights_path": "",
            "segmentation_model_name": "",
            "segmentation_weights_path": "",
            "predicted_label": "",
            "predicted_probability": "",
            "segmentation_mask_path": "",
            "qc_image_path": "",
            "classification_json_path": "",
            "output_directory": output_directory,
            "status": status,
            "error_message": error_message,
        }

    def _process_single_case(
        self,
        input_path: str,
        run_dir: str,
        log_path: str,
        task: str,
        qc_visualization: bool,
        log_to_csv: bool,
        case_directory_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        case_id = _input_case_id(input_path)
        output_case_dir_name = case_directory_name or _case_directory_name(1)
        case_dir = os.path.join(run_dir, output_case_dir_name)
        os.makedirs(case_dir, exist_ok=True)
        row = self._base_log_row(
            case_id=case_id,
            case_directory_name=output_case_dir_name,
            input_path=input_path,
            output_directory=case_dir,
            task=task,
            status="running",
        )

        prepared_case = self.preprocessing_tool.prepare(input_path=input_path, case_id=case_id)
        row["preprocessing_status"] = "success"

        if task in {"segmentation", "all"}:
            segmentation_tool = self._get_segmentation_tool()
            mask = segmentation_tool.predict(prepared_case)
            mask_path = os.path.join(case_dir, "segmentation_mask.nii.gz")
            nib.save(nib.Nifti1Image(mask, prepared_case.affine), mask_path)
            row["segmentation_model_name"] = segmentation_tool.model_name
            row["segmentation_weights_path"] = segmentation_tool.weights_path
            row["segmentation_mask_path"] = mask_path
            if task == "segmentation":
                row["model_name"] = segmentation_tool.model_name
                row["weights_path"] = segmentation_tool.weights_path

            if qc_visualization:
                qc_path = os.path.join(case_dir, "segmentation_qc.png")
                self.visualization_tool.save_segmentation_qc(prepared_case, mask, qc_path)
                row["qc_image_path"] = qc_path

        if task in {"classification", "all"}:
            classification_tool = self._get_classification_tool()
            cls_result = classification_tool.predict(prepared_case)
            json_path = os.path.join(case_dir, "classification.json")
            with open(json_path, "w", encoding="utf-8") as handle:
                json.dump(cls_result, handle, indent=2, sort_keys=True)
            row["classification_model_name"] = classification_tool.model_name
            row["classification_weights_path"] = classification_tool.weights_path
            row["classification_json_path"] = json_path
            row["predicted_label"] = cls_result["predicted_label"]
            row["predicted_probability"] = cls_result["predicted_probability"]
            if task == "classification":
                row["model_name"] = classification_tool.model_name
                row["weights_path"] = classification_tool.weights_path

        if task == "all":
            row["model_name"] = "unet3d_segmentation+classification"
            row["weights_path"] = (
                f"seg={self.settings.segmentation_weights_path};"
                f"cls={self.settings.classification_weights_path}"
            )

        metadata_path = os.path.join(case_dir, "inference_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "case_id": prepared_case.case_id,
                    "case_directory_name": output_case_dir_name,
                    "input_path": prepared_case.input_path,
                    "source_type": prepared_case.source_type,
                    "original_shape": prepared_case.original_shape,
                    "resized_shape": prepared_case.resized_shape,
                    "task": task,
                },
                handle,
                indent=2,
                sort_keys=True,
            )

        row["status"] = "success"
        if log_to_csv:
            self.logger_tool.append(log_path=log_path, row=row)
        print(f"[OK] {case_id}: outputs saved to {case_dir}")
        return row

    def run_single_case(
        self,
        input_path: str,
        task: Optional[str] = None,
        qc_visualization: Optional[bool] = None,
        output_dir: Optional[str] = None,
        log_to_csv: Optional[bool] = None,
    ) -> Dict[str, Any]:
        selected_task = task or self.settings.task
        if selected_task not in {"segmentation", "classification", "all"}:
            raise ValueError(
                "task must be one of: segmentation, classification, all"
            )

        selected_qc = (
            self.settings.qc_visualization
            if qc_visualization is None
            else qc_visualization
        )
        selected_output_dir = output_dir or self.settings.output_dir
        selected_log_to_csv = (
            self.settings.csv_logging if log_to_csv is None else log_to_csv
        )

        run_dir, log_path = self._create_run_context(selected_output_dir)
        row = self._process_single_case(
            input_path=input_path,
            run_dir=run_dir,
            log_path=log_path,
            task=selected_task,
            qc_visualization=selected_qc,
            log_to_csv=selected_log_to_csv,
            case_directory_name=_case_directory_name(1),
        )

        response = dict(row)
        if row["classification_json_path"] and os.path.exists(row["classification_json_path"]):
            with open(row["classification_json_path"], "r", encoding="utf-8") as handle:
                response["classification_result"] = json.load(handle)
        return response

    def run_batch(
        self,
        batch_path: str,
        task: Optional[str] = None,
        qc_visualization: Optional[bool] = None,
        output_dir: Optional[str] = None,
        log_to_csv: Optional[bool] = None,
    ) -> Dict[str, Any]:
        selected_task = task or self.settings.task
        if selected_task not in {"segmentation", "classification", "all"}:
            raise ValueError("task must be one of: segmentation, classification, all")

        selected_qc = (
            self.settings.qc_visualization
            if qc_visualization is None
            else qc_visualization
        )
        selected_output_dir = output_dir or self.settings.output_dir
        selected_log_to_csv = (
            self.settings.csv_logging if log_to_csv is None else log_to_csv
        )

        input_paths = _load_batch_inputs(batch_path)
        run_dir, log_path = self._create_run_context(selected_output_dir)
        rows: List[Dict[str, Any]] = []
        success_count = 0
        failure_count = 0

        for case_index, input_path in enumerate(input_paths, start=1):
            case_dir_name = _case_directory_name(case_index)
            try:
                row = self._process_single_case(
                    input_path=input_path,
                    run_dir=run_dir,
                    log_path=log_path,
                    task=selected_task,
                    qc_visualization=selected_qc,
                    log_to_csv=selected_log_to_csv,
                    case_directory_name=case_dir_name,
                )
                rows.append(row)
                success_count += 1
            except Exception as exc:
                failure_count += 1
                case_id = _input_case_id(input_path)
                case_dir = os.path.join(run_dir, case_dir_name)
                os.makedirs(case_dir, exist_ok=True)
                row = self._base_log_row(
                    case_id=case_id,
                    case_directory_name=case_dir_name,
                    input_path=input_path,
                    output_directory=case_dir,
                    task=selected_task,
                    status="failed",
                    error_message=str(exc),
                )
                rows.append(row)
                if selected_log_to_csv:
                    self.logger_tool.append(log_path=log_path, row=row)

        if failure_count == 0:
            status = "success"
        elif success_count == 0:
            status = "failed"
        else:
            status = "partial_success"

        return {
            "status": status,
            "task": selected_task,
            "batch_input": os.path.expanduser(batch_path),
            "case_count": len(input_paths),
            "success_count": success_count,
            "failure_count": failure_count,
            "run_directory": run_dir,
            "log_path": log_path if selected_log_to_csv else "",
            "cases": rows,
        }

    def _get_classification_tool(self) -> ClassificationTool:
        cache_key = (
            self.settings.classification_weights_path,
            json.dumps(self.settings.classification_class_labels, sort_keys=True),
            self.settings.device,
        )
        if self._classification_tool is None or self._classification_cache_key != cache_key:
            self._classification_tool = ClassificationTool(
                checkpoint_path=self.settings.classification_weights_path,
                model_name=self.settings.classification_model_name,
                cls_targets=self.settings.classification_cls_targets,
                class_labels=self.settings.classification_class_labels,
                device=self.settings.device,
            )
            self._classification_cache_key = cache_key
        return self._classification_tool

    def _get_segmentation_tool(self) -> SegmentationTool:
        cache_key = (
            self.settings.segmentation_weights_path,
            str(self.settings.segmentation_num_classes),
            self.settings.device,
        )
        if self._segmentation_tool is None or self._segmentation_cache_key != cache_key:
            self._segmentation_tool = SegmentationTool(
                checkpoint_path=self.settings.segmentation_weights_path,
                model_name=self.settings.segmentation_model_name,
                num_classes=self.settings.segmentation_num_classes,
                device=self.settings.device,
            )
            self._segmentation_cache_key = cache_key
        return self._segmentation_tool


class MCPInteractiveCLI:
    """Interactive terminal client that talks to the workflow only through MCP."""

    def __init__(self, session: Any) -> None:
        self.session = session

    async def run(self) -> None:
        while True:
            settings = await self._get_settings()
            self._print_menu(settings)
            choice = _prompt("Choose an option", "1")
            if choice == "1":
                await self.process_inputs(settings)
            elif choice == "2":
                await self.change_settings(settings)
            elif choice == "3":
                await self.reset_defaults()
            elif choice == "4":
                await self.view_settings(settings=settings)
            elif choice == "5":
                print("Exiting interactive inference session.")
                return
            else:
                print("Please select 1, 2, 3, 4, or 5.")

    def _print_menu(self, settings: Dict[str, Any]) -> None:
        print("\nInteractive inference workflow")
        self._print_settings(settings, compact=True)
        print("1. Process input")
        print("2. Change settings")
        print("3. Reset to defaults")
        print("4. View current settings")
        print("5. Exit")

    def _print_settings(self, settings: Dict[str, Any], compact: bool = False) -> None:
        if compact:
            print(
                "Defaults: "
                f"mode={settings['input_mode']}, "
                f"task={settings['task']}, "
                f"input={settings['input_format']}, "
                f"qc={'on' if settings['qc_visualization'] else 'off'}, "
                f"logging={'on' if settings['csv_logging'] else 'off'}"
            )
            return

        print("\nCurrent settings")
        for key, value in settings.items():
            print(f"- {key}: {value}")

    async def _call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        result = await self.session.call_tool(name, arguments)
        if getattr(result, "isError", False):
            messages: List[str] = []
            for block in getattr(result, "content", []):
                if getattr(block, "type", None) == "text":
                    messages.append(block.text)
                else:
                    messages.append(json.dumps(block.model_dump(), indent=2, sort_keys=True))
            raise RuntimeError("\n".join(messages) if messages else f"MCP tool failed: {name}")

        if result.structuredContent is not None:
            return _unwrap_mcp_payload(result.structuredContent)

        text_blocks = [
            block.text
            for block in getattr(result, "content", [])
            if getattr(block, "type", None) == "text"
        ]
        if len(text_blocks) == 1:
            try:
                return _unwrap_mcp_payload(json.loads(text_blocks[0]))
            except json.JSONDecodeError:
                return text_blocks[0]
        return text_blocks

    async def _get_settings(self) -> Dict[str, Any]:
        payload = await self._call_tool("get_inference_settings", {})
        if not isinstance(payload, dict):
            raise RuntimeError("MCP server returned invalid settings payload.")
        return payload

    async def view_settings(self, settings: Optional[Dict[str, Any]] = None) -> None:
        self._print_settings(settings or await self._get_settings(), compact=False)

    async def reset_defaults(self) -> None:
        await self._call_tool("reset_inference_settings", {})
        print("Defaults restored.")

    async def change_settings(self, settings: Dict[str, Any]) -> None:
        print("\nUpdate settings. Press Enter to keep the current value.")
        updates = {
            "input_mode": _prompt("Input mode (single/batch)", settings["input_mode"]).lower(),
            "task": _prompt("Task (segmentation/classification/all)", settings["task"]).lower(),
            "input_format": _prompt(
                "Input format hint (nifti/dicom/mixed)", settings["input_format"]
            ).lower(),
            "classification_weights_path": _prompt(
                "Classification weights path",
                settings["classification_weights_path"],
            ),
            "segmentation_weights_path": _prompt(
                "Segmentation weights path",
                settings["segmentation_weights_path"],
            ),
            "qc_visualization": _parse_bool(
                _prompt(
                    "QC visualization (on/off)",
                    "on" if settings["qc_visualization"] else "off",
                ),
                settings["qc_visualization"],
            ),
            "csv_logging": _parse_bool(
                _prompt(
                    "CSV logging (on/off)",
                    "on" if settings["csv_logging"] else "off",
                ),
                settings["csv_logging"],
            ),
            "output_dir": _prompt("Output directory", settings["output_dir"]),
            "device": _prompt("Device", settings["device"]),
        }
        await self._call_tool("update_inference_settings", updates)
        print("Settings updated.")

    async def process_inputs(self, settings: Dict[str, Any]) -> None:
        input_mode = settings["input_mode"]
        if input_mode == "single":
            input_path = _prompt("Case input path")
            case_id = _input_case_id(input_path)
            try:
                result = await self._call_tool("process_case", {"path": input_path})
            except Exception as exc:
                print(f"[FAILED] {case_id}: {exc}")
                return

            print(f"[OK] {case_id}: outputs saved to {result['output_directory']}")
            print(f"Run finished. successes=1, failures=0, outputs={os.path.dirname(result['output_directory'])}")
            return

        if input_mode != "batch":
            raise ValueError(f"Unsupported input mode: {input_mode}")

        batch_path = _prompt("Batch folder or manifest path")
        result = await self._call_tool("process_batch", {"batch_path": batch_path})
        print(
            f"Run finished. successes={result['success_count']}, "
            f"failures={result['failure_count']}, outputs={result['run_directory']}"
        )


def build_mcp_server(
    inference_config_path: str = DEFAULT_INFERENCE_CONFIG_PATH,
    device: str = "auto",
) -> Any:
    """Create a protocol-level MCP server around the existing inference workflow."""
    if FastMCP is None:
        raise ImportError(
            'The MCP Python SDK is not installed. Install it with: pip install "mcp[cli]"'
        )

    settings = build_default_settings(
        inference_config_path=inference_config_path,
        device=device,
    )
    workflow = InteractiveInferenceWorkflow(settings=settings)
    mcp = FastMCP(
        name="medical-imaging-inference",
        instructions=(
            "Tool-based medical imaging inference server for preprocessing, segmentation, "
            "classification, QC visualization, and structured result logging."
        ),
        stateless_http=True,
        json_response=True,
    )

    def _run_via_mcp_tool(callback: Any) -> Any:
        with contextlib.redirect_stdout(io.StringIO()):
            return callback()

    @mcp.tool()
    def get_inference_settings() -> Dict[str, Any]:
        """Return the current inference defaults exposed by this server."""
        return workflow.get_settings_dict()

    @mcp.tool()
    def update_inference_settings(
        input_mode: Optional[str] = None,
        task: Optional[str] = None,
        input_format: Optional[str] = None,
        classification_weights_path: Optional[str] = None,
        segmentation_weights_path: Optional[str] = None,
        qc_visualization: Optional[bool] = None,
        csv_logging: Optional[bool] = None,
        output_dir: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update the inference defaults used by subsequent MCP tool calls."""
        return workflow.update_settings(
            input_mode=input_mode,
            task=task,
            input_format=input_format,
            classification_weights_path=classification_weights_path,
            segmentation_weights_path=segmentation_weights_path,
            qc_visualization=qc_visualization,
            csv_logging=csv_logging,
            output_dir=output_dir,
            device=device,
        )

    @mcp.tool()
    def reset_inference_settings() -> Dict[str, Any]:
        """Reset inference defaults from the config file."""
        with contextlib.redirect_stdout(io.StringIO()):
            return workflow.reset_defaults()

    def _process_case_impl(
        path: str,
        task: str = "all",
        qc_visualization: Optional[bool] = None,
        output_dir: Optional[str] = None,
        log_to_csv: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Run preprocessing and selected downstream inference tasks for one case."""
        return _run_via_mcp_tool(
            lambda: workflow.run_single_case(
                input_path=path,
                task=task,
                qc_visualization=qc_visualization,
                output_dir=output_dir,
                log_to_csv=log_to_csv,
            )
        )

    @mcp.tool()
    def process_case(
        path: str,
        task: str = "all",
        qc_visualization: Optional[bool] = None,
        output_dir: Optional[str] = None,
        log_to_csv: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Run preprocessing and selected downstream inference tasks for one case."""
        return _process_case_impl(
            path=path,
            task=task,
            qc_visualization=qc_visualization,
            output_dir=output_dir,
            log_to_csv=log_to_csv,
        )

    @mcp.tool()
    def process_oai_case(
        path: str,
        task: str = "all",
        qc_visualization: Optional[bool] = None,
        output_dir: Optional[str] = None,
        log_to_csv: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Backward-compatible alias for process_case."""
        return _process_case_impl(
            path=path,
            task=task,
            qc_visualization=qc_visualization,
            output_dir=output_dir,
            log_to_csv=log_to_csv,
        )

    def _process_batch_impl(
        batch_path: str,
        task: str = "all",
        qc_visualization: Optional[bool] = None,
        output_dir: Optional[str] = None,
        log_to_csv: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Run batch inference from a folder or manifest and continue on per-case failures."""
        return _run_via_mcp_tool(
            lambda: workflow.run_batch(
                batch_path=batch_path,
                task=task,
                qc_visualization=qc_visualization,
                output_dir=output_dir,
                log_to_csv=log_to_csv,
            )
        )

    @mcp.tool()
    def process_batch(
        batch_path: str,
        task: str = "all",
        qc_visualization: Optional[bool] = None,
        output_dir: Optional[str] = None,
        log_to_csv: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Run batch inference from a folder or manifest and continue on per-case failures."""
        return _process_batch_impl(
            batch_path=batch_path,
            task=task,
            qc_visualization=qc_visualization,
            output_dir=output_dir,
            log_to_csv=log_to_csv,
        )

    @mcp.tool()
    def process_oai_batch(
        batch_path: str,
        task: str = "all",
        qc_visualization: Optional[bool] = None,
        output_dir: Optional[str] = None,
        log_to_csv: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Backward-compatible alias for process_batch."""
        return _process_batch_impl(
            batch_path=batch_path,
            task=task,
            qc_visualization=qc_visualization,
            output_dir=output_dir,
            log_to_csv=log_to_csv,
        )

    return mcp


async def run_interactive_cli_via_mcp(
    inference_config_path: str,
    device: str,
    host: str,
    port: int,
    mcp_path: str,
) -> None:
    if ClientSession is None or streamable_http_client is None:
        raise ImportError(
            'The MCP Python SDK is not installed. Install it with: pip install "mcp[cli]"'
        )

    if port <= 0:
        raise ValueError("port must be a positive integer when starting the background MCP server")

    server_log = tempfile.NamedTemporaryFile(
        mode="w+",
        encoding="utf-8",
        delete=False,
        prefix="inference_mcp_server_",
        suffix=".log",
    )
    server_log_path = server_log.name
    server_process: Optional[subprocess.Popen[str]] = None
    normalized_mcp_path = _normalize_mcp_path(mcp_path)
    server_url = _build_mcp_server_url(host=host, port=port, mcp_path=normalized_mcp_path)
    client_url = _build_mcp_server_url(
        host=_mcp_client_host(host),
        port=port,
        mcp_path=normalized_mcp_path,
    )

    try:
        server_process = subprocess.Popen(
            [
                sys.executable,
                os.path.abspath(__file__),
                "--inference-config",
                inference_config_path,
                "--device",
                device,
                "--serve-mcp",
                "--host",
                host,
                "--port",
                str(port),
                "--mcp-path",
                normalized_mcp_path,
            ],
            cwd=os.getcwd(),
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _wait_for_background_mcp_server(
            process=server_process,
            host=_mcp_client_host(host),
            port=port,
            mcp_path=normalized_mcp_path,
            log_path=server_log_path,
        )
        print(f"Background MCP server ready at {server_url}")

        async with streamable_http_client(client_url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                cli = MCPInteractiveCLI(session)
                await cli.run()
    finally:
        if server_process is not None:
            _shutdown_background_mcp_server(server_process)
        server_log.close()
        if os.path.exists(server_log_path):
            os.remove(server_log_path)


def _normalize_mcp_path(mcp_path: str) -> str:
    if not mcp_path.startswith("/"):
        return f"/{mcp_path}"
    return mcp_path


def _mcp_client_host(host: str) -> str:
    if host in {"0.0.0.0", "::", "[::]"}:
        return "127.0.0.1"
    return host


def _build_mcp_server_url(host: str, port: int, mcp_path: str) -> str:
    return f"http://{host}:{port}{_normalize_mcp_path(mcp_path)}"


def _read_log_tail(log_path: str, max_chars: int = 4000) -> str:
    if not os.path.exists(log_path):
        return ""

    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        contents = handle.read()
    return contents[-max_chars:].strip()


def _wait_for_background_mcp_server(
    process: subprocess.Popen[str],
    host: str,
    port: int,
    mcp_path: str,
    log_path: str,
    timeout_seconds: float = 30.0,
) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Optional[str] = None

    while time.time() < deadline:
        if process.poll() is not None:
            break
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError as exc:
            last_error = str(exc)
            time.sleep(0.2)

    details = _read_log_tail(log_path)
    message = (
        "Background MCP server failed to start at "
        f"{_build_mcp_server_url(host=host, port=port, mcp_path=mcp_path)}."
    )
    if process.poll() is not None:
        message += f" Exit code: {process.returncode}."
    elif last_error:
        message += f" Last connection error: {last_error}."
    if details:
        message += f"\nServer log tail:\n{details}"
    raise RuntimeError(message)


def _shutdown_background_mcp_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    try:
        process.send_signal(signal.SIGINT)
        process.wait(timeout=10)
    except (ProcessLookupError, ValueError):
        return
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def build_default_settings(
    inference_config_path: str = DEFAULT_INFERENCE_CONFIG_PATH,
    device: str = "auto",
) -> SessionSettings:
    inference_cfg = _load_config(inference_config_path)
    cfg_base = os.path.dirname(inference_config_path)
    cfg_device = str(inference_cfg.get("device", "auto"))
    classification_class_labels = inference_cfg.get("classification_class_labels") or {}

    return SessionSettings(
        input_mode=str(inference_cfg.get("input_mode", "single")),
        task=str(inference_cfg.get("task", "all")),
        input_format=str(inference_cfg.get("input_format", "nifti")),
        classification_weights_path=_resolve_path(
            inference_cfg.get("classification_weights_path"), cfg_base
        )
        or DEFAULT_CLASSIFICATION_WEIGHTS,
        segmentation_weights_path=_resolve_path(
            inference_cfg.get("segmentation_weights_path"), cfg_base
        )
        or DEFAULT_SEGMENTATION_WEIGHTS,
        classification_model_name=str(
            inference_cfg.get("classification_model_name", "unet3d_classification")
        ),
        segmentation_model_name=str(
            inference_cfg.get("segmentation_model_name", "unet3d_segmentation")
        ),
        classification_cls_targets=list(
            inference_cfg.get("classification_cls_targets", ["V00COHORT", "gender"])
        ),
        classification_class_labels={
            key: list(value)
            for key, value in dict(classification_class_labels).items()
        },
        segmentation_num_classes=int(inference_cfg.get("segmentation_num_classes", 7)),
        preprocessing_resize_to=tuple(
            int(value)
            for value in inference_cfg.get("preprocessing_resize_to", [40, 224, 224])
        ),
        qc_visualization=bool(inference_cfg.get("qc_visualization", False)),
        csv_logging=bool(inference_cfg.get("csv_logging", True)),
        output_dir=_resolve_path(inference_cfg.get("output_dir"), cfg_base) or DEFAULT_OUTPUT_DIR,
        device=device if device != "auto" else cfg_device,
        inference_config_path=inference_config_path,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Persistent interactive CLI workflow for segmentation and classification inference."
    )
    parser.add_argument(
        "--inference-config",
        default=DEFAULT_INFERENCE_CONFIG_PATH,
        help="Path to the interactive inference config.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, for example auto, cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--serve-mcp",
        action="store_true",
        help="Start the MCP server instead of the interactive CLI session.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host used by the MCP server when --serve-mcp is enabled.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port used by the MCP server mode.",
    )
    parser.add_argument(
        "--mcp-path",
        default="/mcp",
        help="HTTP path used by the MCP server when --serve-mcp is enabled.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    if args.serve_mcp:
        mcp = build_mcp_server(
            inference_config_path=args.inference_config,
            device=args.device,
        )
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.settings.streamable_http_path = args.mcp_path
        print(
            f"Starting MCP inference server at http://{args.host}:{args.port}{args.mcp_path}"
        )
        mcp.run(transport="streamable-http")
        return

    asyncio.run(
        run_interactive_cli_via_mcp(
            inference_config_path=args.inference_config,
            device=args.device,
            host=args.host,
            port=args.port,
            mcp_path=args.mcp_path,
        )
    )


if __name__ == "__main__":
    main()
