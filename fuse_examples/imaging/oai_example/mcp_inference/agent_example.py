# mypy: python_version=3.10
"""
Runnable MCP agent example for the OAI inference workflow.

The agent is deliberately deterministic: it observes the MCP tool catalogue and
settings, plans a single-case or batch action, invokes the selected tool, and
writes a structured report. It therefore runs without sending medical images or
metadata to an external LLM service.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import nibabel as nib
import numpy as np

try:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
except ImportError:
    ClientSession = None
    streamable_http_client = None

from inference_utils import (
    DEFAULT_INFERENCE_CONFIG_PATH,
    REPO_ROOT,
    _build_mcp_server_url,
    _mcp_client_host,
    _normalize_mcp_path,
    _unwrap_mcp_payload,
)


class MCPInferenceAgent:
    """A bounded observe-plan-act agent backed exclusively by MCP tools."""

    def __init__(self, session: Any, server_url: str) -> None:
        self.session = session
        self.server_url = server_url
        self.events: List[Dict[str, Any]] = []
        self.available_tools: set[str] = set()

    def _record(self, phase: str, message: str, **details: Any) -> None:
        event = {"phase": phase, "message": message, **details}
        self.events.append(event)
        print(f"Agent {phase}: {message}")

    async def _call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        if name not in self.available_tools:
            raise RuntimeError(
                f"MCP tool '{name}' is unavailable. "
                f"Available tools: {sorted(self.available_tools)}"
            )

        result = await self.session.call_tool(name, arguments)
        if getattr(result, "isError", False):
            messages = [
                block.text
                for block in getattr(result, "content", [])
                if getattr(block, "type", None) == "text"
            ]
            raise RuntimeError(
                "\n".join(messages) if messages else f"MCP tool failed: {name}"
            )

        structured_content = getattr(result, "structuredContent", None)
        if structured_content is not None:
            return _unwrap_mcp_payload(structured_content)

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

    async def observe(self) -> Dict[str, Any]:
        """Discover server capabilities and current inference settings."""
        catalogue = await self.session.list_tools()
        self.available_tools = {tool.name for tool in catalogue.tools}
        required = {
            "get_inference_settings",
            "update_inference_settings",
            "process_case",
            "process_batch",
        }
        missing = sorted(required - self.available_tools)
        if missing:
            raise RuntimeError(f"MCP server is missing required tools: {missing}")

        settings = await self._call_tool("get_inference_settings", {})
        if not isinstance(settings, dict):
            raise RuntimeError("MCP server returned invalid inference settings.")
        self._record(
            "observation",
            f"discovered {len(self.available_tools)} tools and current settings",
            tools=sorted(self.available_tools),
        )
        return settings

    @staticmethod
    def _validate_weights(settings: Dict[str, Any], task: str) -> None:
        required_paths: List[Tuple[str, str]] = []
        if task in {"classification", "all"}:
            required_paths.append(
                (
                    "classification",
                    str(settings.get("classification_weights_path", "")),
                )
            )
        if task in {"segmentation", "all"}:
            required_paths.append(
                (
                    "segmentation",
                    str(settings.get("segmentation_weights_path", "")),
                )
            )

        missing = [
            f"{name}: {path}"
            for name, path in required_paths
            if not os.path.isfile(path)
        ]
        if missing:
            raise FileNotFoundError(
                "Required checkpoint(s) not found. Pass the corresponding "
                "--classification-weights or --segmentation-weights argument: "
                + "; ".join(missing)
            )

    async def configure(
        self,
        *,
        input_count: int,
        task: str,
        classification_weights: str | None,
        segmentation_weights: str | None,
        output_dir: str | None,
        device: str,
        qc_visualization: bool,
        csv_logging: bool,
    ) -> Dict[str, Any]:
        """Configure model and run settings through the MCP server."""
        updates: Dict[str, Any] = {
            "input_mode": "single" if input_count == 1 else "batch",
            "task": task,
            "device": device,
            "qc_visualization": qc_visualization,
            "csv_logging": csv_logging,
        }
        if classification_weights is not None:
            updates["classification_weights_path"] = os.path.abspath(
                os.path.expanduser(classification_weights)
            )
        if segmentation_weights is not None:
            updates["segmentation_weights_path"] = os.path.abspath(
                os.path.expanduser(segmentation_weights)
            )
        if output_dir is not None:
            updates["output_dir"] = os.path.abspath(os.path.expanduser(output_dir))

        settings = await self._call_tool("update_inference_settings", updates)
        if not isinstance(settings, dict):
            raise RuntimeError("MCP server returned invalid updated settings.")
        self._validate_weights(settings, task)
        self._record(
            "configuration",
            f"configured task={task}, mode={updates['input_mode']}, device={device}",
        )
        return settings

    @staticmethod
    def _build_case_summary(row: Dict[str, Any]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "case_id": row.get("case_id"),
            "status": row.get("status"),
            "error_message": row.get("error_message", ""),
            "output_directory": row.get("output_directory", ""),
            "classification_json_path": row.get("classification_json_path", ""),
            "segmentation_mask_path": row.get("segmentation_mask_path", ""),
            "qc_image_path": row.get("qc_image_path", ""),
        }

        classification_path = str(row.get("classification_json_path", ""))
        if classification_path and os.path.isfile(classification_path):
            with open(classification_path, encoding="utf-8") as handle:
                classification = json.load(handle)
            summary["classification"] = {
                "primary_target": classification.get("primary_target"),
                "predicted_label": classification.get("predicted_label"),
                "predicted_probability": classification.get("predicted_probability"),
                "targets": classification.get("targets", {}),
            }

        segmentation_path = str(row.get("segmentation_mask_path", ""))
        if segmentation_path and os.path.isfile(segmentation_path):
            mask = np.asanyarray(nib.load(segmentation_path).dataobj)
            labels, counts = np.unique(mask, return_counts=True)
            summary["segmentation"] = {
                "shape": [int(value) for value in mask.shape],
                "label_voxel_counts": {
                    str(int(label)): int(count) for label, count in zip(labels, counts)
                },
            }

        return summary

    def _build_report(
        self,
        plan: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        rows = result.get("cases")
        if not isinstance(rows, list):
            rows = [result]

        run_directory = str(result.get("run_directory", ""))
        if not run_directory and rows:
            output_directory = str(rows[0].get("output_directory", ""))
            run_directory = os.path.dirname(output_directory)

        return {
            "agent_type": "deterministic_mcp_tool_agent",
            "server_url": self.server_url,
            "observed_tools": sorted(self.available_tools),
            "plan": plan,
            "status": result.get("status"),
            "task": result.get("task", plan["arguments"]["task"]),
            "case_count": result.get("case_count", len(rows)),
            "success_count": result.get(
                "success_count",
                sum(row.get("status") == "success" for row in rows),
            ),
            "failure_count": result.get(
                "failure_count",
                sum(row.get("status") != "success" for row in rows),
            ),
            "run_directory": run_directory,
            "inference_log_path": result.get("log_path", ""),
            "cases": [self._build_case_summary(row) for row in rows],
            "events": self.events,
        }

    async def run(
        self,
        *,
        inputs: Sequence[str],
        task: str,
        classification_weights: str | None,
        segmentation_weights: str | None,
        output_dir: str | None,
        device: str,
        qc_visualization: bool,
        csv_logging: bool,
    ) -> Dict[str, Any]:
        """Observe, plan, execute, and report one inference request."""
        await self.observe()
        await self.configure(
            input_count=len(inputs),
            task=task,
            classification_weights=classification_weights,
            segmentation_weights=segmentation_weights,
            output_dir=output_dir,
            device=device,
            qc_visualization=qc_visualization,
            csv_logging=csv_logging,
        )

        common_arguments: Dict[str, Any] = {
            "task": task,
            "qc_visualization": qc_visualization,
            "log_to_csv": csv_logging,
        }
        if output_dir is not None:
            common_arguments["output_dir"] = os.path.abspath(
                os.path.expanduser(output_dir)
            )

        if len(inputs) == 1:
            arguments = {"path": inputs[0], **common_arguments}
            plan = {"tool": "process_case", "arguments": arguments}
            self._record("plan", f"selected process_case for {inputs[0]}")
            self._record("action", "calling process_case")
            result = await self._call_tool("process_case", arguments)
        else:
            with tempfile.TemporaryDirectory(prefix="oai_mcp_agent_") as temp_dir:
                manifest_path = os.path.join(temp_dir, "inputs.jsonl")
                with open(manifest_path, "w", encoding="utf-8") as handle:
                    for input_path in inputs:
                        handle.write(json.dumps({"path": input_path}) + "\n")
                arguments = {"batch_path": manifest_path, **common_arguments}
                plan = {
                    "tool": "process_batch",
                    "arguments": {**arguments, "inputs": list(inputs)},
                }
                self._record("plan", f"selected process_batch for {len(inputs)} cases")
                self._record("action", "calling process_batch")
                result = await self._call_tool("process_batch", arguments)

        if not isinstance(result, dict):
            raise RuntimeError("MCP inference tool returned an invalid result.")
        report = self._build_report(plan=plan, result=result)
        self._record(
            "result",
            f"status={report['status']}, successes={report['success_count']}, "
            f"failures={report['failure_count']}",
        )
        report["events"] = self.events

        run_directory = report["run_directory"]
        if run_directory:
            report_path = os.path.join(run_directory, "agent_report.json")
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
            report["agent_report_path"] = report_path
        return report


def _select_inputs(
    explicit_inputs: Sequence[str],
    sample_dir: str | None,
    limit: int,
) -> List[str]:
    """Resolve explicit inputs and optionally select representative NIfTI cases."""
    if limit < 1:
        raise ValueError("--limit must be a positive integer")

    selected = [os.path.abspath(os.path.expanduser(path)) for path in explicit_inputs]
    if sample_dir is not None:
        sample_root = Path(sample_dir).expanduser().resolve()
        if not sample_root.is_dir():
            raise NotADirectoryError(f"Sample directory not found: {sample_root}")
        candidates = sorted(
            path
            for path in (*sample_root.glob("*.nii"), *sample_root.glob("*.nii.gz"))
            if "_cartilage" not in path.name
        )
        selected.extend(str(path) for path in candidates[:limit])

    selected = list(dict.fromkeys(selected))
    if not selected:
        raise ValueError("Provide at least one --input or a --sample-dir.")
    missing = [path for path in selected if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Input path(s) not found: {missing}")
    return selected


def _reserve_port(host: str) -> int:
    bind_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((bind_host, 0))
        return int(listener.getsockname()[1])


def _wait_for_server(
    process: subprocess.Popen[str],
    host: str,
    port: int,
    log_path: str,
    timeout_seconds: float = 45.0,
) -> None:
    deadline = time.time() + timeout_seconds
    connect_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    while time.time() < deadline:
        if process.poll() is not None:
            break
        try:
            with socket.create_connection((connect_host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.2)

    details = ""
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            details = handle.read()[-4000:].strip()
    raise RuntimeError(
        f"MCP server failed to start on {connect_host}:{port}."
        + (f"\nServer log tail:\n{details}" if details else "")
    )


def _stop_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        process.send_signal(signal.SIGINT)
        process.wait(timeout=10)
    except (ProcessLookupError, ValueError):
        return
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a deterministic agent against the OAI MCP inference server."
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="NIfTI file or DICOM directory. Repeat for multiple cases.",
    )
    parser.add_argument(
        "--sample-dir",
        help="Select sorted NIfTI inputs from this folder (cartilage masks excluded).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum number selected from --sample-dir (default: 3).",
    )
    parser.add_argument(
        "--task",
        choices=["segmentation", "classification", "all"],
        default="all",
    )
    parser.add_argument("--classification-weights")
    parser.add_argument("--segmentation-weights")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--no-qc", action="store_true", help="Disable segmentation QC images."
    )
    parser.add_argument(
        "--no-csv-log", action="store_true", help="Disable inference_log.csv."
    )
    parser.add_argument(
        "--server-url",
        help="Use an existing Streamable HTTP MCP server instead of starting one.",
    )
    parser.add_argument(
        "--inference-config",
        default=DEFAULT_INFERENCE_CONFIG_PATH,
        help="Config used when starting the local MCP server.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Local MCP port; 0 selects an available port.",
    )
    parser.add_argument("--mcp-path", default="/mcp")
    return parser


async def _run_agent(args: argparse.Namespace) -> Dict[str, Any]:
    if ClientSession is None or streamable_http_client is None:
        raise ImportError(
            "The MCP Python SDK is not installed. Install it with: "
            'pip install "mcp[cli]"'
        )

    inputs = _select_inputs(args.input, args.sample_dir, args.limit)
    process: subprocess.Popen[str] | None = None
    server_log = None
    server_log_path = ""

    if args.server_url:
        server_url = args.server_url
    else:
        port = args.port or _reserve_port(args.host)
        normalized_path = _normalize_mcp_path(args.mcp_path)
        server_url = _build_mcp_server_url(
            _mcp_client_host(args.host), port, normalized_path
        )
        server_log = tempfile.NamedTemporaryFile(
            mode="w+",
            encoding="utf-8",
            delete=False,
            prefix="oai_agent_mcp_server_",
            suffix=".log",
        )
        server_log_path = server_log.name
        process = subprocess.Popen(
            [
                sys.executable,
                os.path.join(os.path.dirname(__file__), "inference_cli.py"),
                "--inference-config",
                os.path.abspath(args.inference_config),
                "--device",
                args.device,
                "--serve-mcp",
                "--host",
                args.host,
                "--port",
                str(port),
                "--mcp-path",
                normalized_path,
            ],
            cwd=REPO_ROOT,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            _wait_for_server(process, args.host, port, server_log_path)
        except Exception:
            _stop_server(process)
            server_log.close()
            if os.path.exists(server_log_path):
                os.remove(server_log_path)
            raise
        print(f"Agent server: started temporary MCP server at {server_url}")

    try:
        async with streamable_http_client(server_url) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                agent = MCPInferenceAgent(session=session, server_url=server_url)
                return await agent.run(
                    inputs=inputs,
                    task=args.task,
                    classification_weights=args.classification_weights,
                    segmentation_weights=args.segmentation_weights,
                    output_dir=args.output_dir,
                    device=args.device,
                    qc_visualization=not args.no_qc,
                    csv_logging=not args.no_csv_log,
                )
    finally:
        if process is not None:
            _stop_server(process)
        if server_log is not None:
            server_log.close()
        if server_log_path and os.path.exists(server_log_path):
            os.remove(server_log_path)


def main() -> None:
    args = _build_arg_parser().parse_args()
    report = asyncio.run(_run_agent(args))
    print("\nAgent report")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
