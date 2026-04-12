# mypy: python_version=3.10
"""Tool implementations and main inference orchestrator."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from inference_utils import (
    PreparedCase,
    ResultLoggerTool,
    SessionSettings,
    _case_directory_name,
    _input_case_id,
    _jsonable_value,
    _resolve_device,
)


class PreprocessingTool:
    """Load, normalize, and resize MRI volumes for downstream inference."""

    def __init__(self, resize_to: Sequence[int]):
        self.resize_to = tuple(int(value) for value in resize_to)

    def prepare(self, input_path: str, case_id: str | None = None) -> PreparedCase:
        """Prepare case from input path."""
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
        elif os.path.isfile(resolved_path) and resolved_path.endswith(
            (".nii", ".nii.gz")
        ):
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
        class_labels: Dict[str, Sequence[Any]] | None,
        device: str,
    ) -> None:
        from inference_utils import (
            _extract_checkpoint_state_dict,
            _infer_head_output_dims,
        )

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
            self.classes_by_target[target] = [
                _jsonable_value(label) for label in labels
            ]

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
        self.model = ModelMultiHead(
            conv_inputs=(("img", 1),), backbone=backbone, heads=heads
        )

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

        missing_keys, unexpected_keys = self.model.load_state_dict(
            state_dict, strict=False
        )
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
            print(
                "Ignoring unexpected checkpoint keys:", ", ".join(unexpected_keys[:5])
            )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, prepared_case: PreparedCase) -> Dict[str, Any]:
        """Run classification prediction."""
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
            logits = (
                batch_dict[f"model.logits.head_{target}"][0].detach().cpu().tolist()
            )
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
        import torch.nn as nn

        from fuse.dl.models import ModelMultiHead
        from fuse.dl.models.backbones.backbone_unet3d import UNet3D
        from fuse.dl.models.heads.head_dense_segmentation import HeadDenseSegmentation

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
        self.model = ModelMultiHead(
            conv_inputs=(("img", 1),), backbone=backbone, heads=heads
        )
        self._load_checkpoint()
        self.model.to(self.device)
        self.model.eval()

    def _load_checkpoint(self) -> None:
        """Load checkpoint into model."""
        from inference_utils import _extract_checkpoint_state_dict

        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"Segmentation checkpoint not found: {self.weights_path}"
            )

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
        """Run segmentation prediction."""
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
        """Save segmentation QC image."""
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


class MCPInferenceEngine:
    """Session controller for tool-based downstream inference orchestration."""

    def __init__(self, settings: SessionSettings) -> None:
        self.settings = settings
        self.preprocessing_tool = PreprocessingTool(
            resize_to=self.settings.preprocessing_resize_to
        )
        self.visualization_tool = VisualizationTool()
        self.logger_tool = ResultLoggerTool()
        self._classification_tool: ClassificationTool | None = None
        self._segmentation_tool: SegmentationTool | None = None
        self._classification_cache_key: Tuple[str, str, str] | None = None
        self._segmentation_cache_key: Tuple[str, str, str] | None = None
        self._shutdown_requested = False

    def get_settings_dict(self) -> Dict[str, Any]:
        """Get current settings as dictionary."""
        return asdict(self.settings)

    def request_shutdown(self) -> None:
        """Signal that inference should shut down gracefully."""
        self._shutdown_requested = True

    def _invalidate_model_caches(self) -> None:
        """Invalidate cached models."""
        self._classification_tool = None
        self._segmentation_tool = None
        self._classification_cache_key = None
        self._segmentation_cache_key = None

    def update_settings(
        self,
        *,
        input_mode: str | None = None,
        task: str | None = None,
        input_format: str | None = None,
        classification_weights_path: str | None = None,
        segmentation_weights_path: str | None = None,
        qc_visualization: bool | None = None,
        csv_logging: bool | None = None,
        output_dir: str | None = None,
        device: str | None = None,
    ) -> Dict[str, Any]:
        """Update settings."""
        if input_mode is not None:
            normalized_input_mode = input_mode.lower()
            if normalized_input_mode not in {"single", "batch"}:
                raise ValueError("input_mode must be one of: single, batch")
            self.settings.input_mode = normalized_input_mode

        if task is not None:
            normalized_task = task.lower()
            if normalized_task not in {"segmentation", "classification", "all"}:
                raise ValueError(
                    "task must be one of: segmentation, classification, all"
                )
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

    def _create_run_context(self, output_dir: str) -> Tuple[str, str]:
        """Create run directory and log path."""
        from datetime import datetime

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
        """Create base log row."""
        from datetime import datetime

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
        case_directory_name: str | None = None,
    ) -> Dict[str, Any]:
        """Process a single case."""
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

        prepared_case = self.preprocessing_tool.prepare(
            input_path=input_path, case_id=case_id
        )
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
                self.visualization_tool.save_segmentation_qc(
                    prepared_case, mask, qc_path
                )
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
        task: str | None = None,
        qc_visualization: bool | None = None,
        output_dir: str | None = None,
        log_to_csv: bool | None = None,
    ) -> Dict[str, Any]:
        """Run inference on single case."""
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
        if row["classification_json_path"] and os.path.exists(
            row["classification_json_path"]
        ):
            with open(row["classification_json_path"], encoding="utf-8") as handle:
                response["classification_result"] = json.load(handle)
        return response

    def run_batch(
        self,
        batch_path: str,
        task: str | None = None,
        qc_visualization: bool | None = None,
        output_dir: str | None = None,
        log_to_csv: bool | None = None,
    ) -> Dict[str, Any]:
        """Run inference on batch."""
        from inference_utils import _load_batch_inputs

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
            if self._shutdown_requested:
                print(
                    f"\nBatch processing interrupted. Processed {success_count + failure_count}/{len(input_paths)} cases."
                )
                break

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

        if failure_count == 0 and not self._shutdown_requested:
            status = "success"
        elif success_count == 0 or self._shutdown_requested:
            status = "failed" if success_count == 0 else "interrupted"
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
        """Get or create classification tool."""
        cache_key = (
            self.settings.classification_weights_path,
            json.dumps(self.settings.classification_class_labels, sort_keys=True),
            self.settings.device,
        )
        if (
            self._classification_tool is None
            or self._classification_cache_key != cache_key
        ):
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
        """Get or create segmentation tool."""
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
