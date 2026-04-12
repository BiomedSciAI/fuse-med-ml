# mypy: python_version=3.10
"""Utility functions, constants, and data models for inference pipeline."""
from __future__ import annotations

import ast
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

OAI_EXAMPLE_DIR = os.path.dirname(BASE_DIR)
DEFAULT_INFERENCE_CONFIG_PATH = os.path.join(BASE_DIR, "inference_config.yaml")
DEFAULT_OUTPUT_DIR = os.path.join(OAI_EXAMPLE_DIR, "outputs", "mcp_inference")
DEFAULT_CLASSIFICATION_WEIGHTS = os.path.join(
    BASE_DIR, "weights", "classification_model.ckpt"
)
DEFAULT_SEGMENTATION_WEIGHTS = os.path.join(
    BASE_DIR, "weights", "segmentation_model.ckpt"
)
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


# ============================================================================
# Utility Functions
# ============================================================================


def _jsonable_value(value: Any) -> Any:
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _unwrap_mcp_payload(payload: Any) -> Any:
    """Unwrap MCP-wrapped payload."""
    if isinstance(payload, dict) and set(payload.keys()) == {"result"}:
        return payload["result"]
    return payload


def _resolve_device(requested_device: str) -> torch.device:
    """Resolve device string to torch.device, with fallback to CPU."""
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device(requested_device)


def _parse_config_value(value: str) -> Any:
    """Parse config value from string representation."""
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
    """Load configuration from file."""
    config: Dict[str, Any] = {}
    if not os.path.exists(config_path):
        return config

    with open(config_path, encoding="utf-8") as handle:
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
    """Extract possible state dict candidates from checkpoint."""
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
    """Choose best matching state dict from candidates."""
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
    """Extract and match state dict from checkpoint."""
    for candidate in _state_dict_candidates(checkpoint):
        chosen = _choose_state_dict(candidate, model_keys)
        if chosen:
            return chosen

    raise ValueError(
        "Unsupported checkpoint format. Expected a Lightning checkpoint or a raw state dict."
    )


def _infer_head_output_dims(
    checkpoint: Any, cls_targets: Sequence[str]
) -> Dict[str, int]:
    """Infer classification head output dimensions from checkpoint."""
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
                target
                for target in cls_targets
                if target not in available_named_targets
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


def _resolve_path(path: str | None, base_dir: str) -> str | None:
    """Resolve path relative to base directory."""
    if path is None:
        return None
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def _input_case_id(input_path: str) -> str:
    """Extract case ID from input path."""
    normalized = os.path.normpath(input_path)
    if os.path.isdir(normalized):
        return os.path.basename(normalized)
    filename = os.path.basename(normalized)
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    return os.path.splitext(filename)[0]


def _case_directory_name(case_index: int) -> str:
    """Generate case directory name from index."""
    if case_index < 1:
        raise ValueError("case_index must be a positive integer")
    return f"case_{case_index:04d}"


def _parse_bool(text: str, default: bool) -> bool:
    """Parse boolean from user input."""
    value = text.strip().lower()
    if value == "":
        return default
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected yes/no style input, got: {text}")


def _prompt(prompt_text: str, default: str | None = None) -> str:
    """Prompt user for input."""
    suffix = f" [{default}]" if default not in (None, "") else ""
    value = input(f"{prompt_text}{suffix}: ").strip()
    if value == "" and default is not None:
        return default
    return value


def _load_batch_inputs(batch_path: str) -> List[str]:
    """Load batch input paths from folder or manifest."""
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
        with open(batch_path, encoding="utf-8") as handle:
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
                return [
                    _resolve_manifest_path(path)
                    for path in df[column].dropna().astype(str)
                ]
        if len(df.columns) == 0:
            raise ValueError(f"No columns found in {batch_path}")
        return [
            _resolve_manifest_path(path) for path in df.iloc[:, 0].dropna().astype(str)
        ]

    raise ValueError(
        "Batch input must be a directory, .csv, .tsv, .txt, or .jsonl manifest."
    )


def _normalize_mcp_path(mcp_path: str) -> str:
    """Ensure MCP path starts with /."""
    if not mcp_path.startswith("/"):
        return f"/{mcp_path}"
    return mcp_path


def _mcp_client_host(host: str) -> str:
    """Convert server host to client-accessible host."""
    if host in {"0.0.0.0", "::", "[::]"}:
        return "127.0.0.1"
    return host


def _build_mcp_server_url(host: str, port: int, mcp_path: str) -> str:
    """Build MCP server URL."""
    return f"http://{host}:{port}{_normalize_mcp_path(mcp_path)}"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class SessionSettings:
    """Inference session settings."""

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
    """Preprocessed case ready for inference."""

    case_id: str
    input_path: str
    source_type: str
    original_shape: Tuple[int, ...]
    resized_shape: Tuple[int, ...]
    tensor: torch.Tensor
    resized_image: np.ndarray
    affine: np.ndarray


class ResultLoggerTool:
    """Append per-case run metadata to CSV."""

    def append(self, log_path: str, row: Dict[str, Any]) -> None:
        """Append row to CSV log."""
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_exists = os.path.exists(log_path)
        with open(log_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LOG_FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in LOG_FIELDNAMES})
