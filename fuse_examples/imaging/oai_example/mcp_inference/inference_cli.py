# mypy: python_version=3.10
from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict

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

from inference_tools import MCPInferenceEngine
from inference_utils import (
    DEFAULT_CLASSIFICATION_WEIGHTS,
    DEFAULT_INFERENCE_CONFIG_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEGMENTATION_WEIGHTS,
    SessionSettings,
    _build_mcp_server_url,
    _input_case_id,
    _load_config,
    _mcp_client_host,
    _normalize_mcp_path,
    _parse_bool,
    _prompt,
    _resolve_path,
    _unwrap_mcp_payload,
)


class MCPInteractiveCLI:
    """Interactive terminal client that talks to the workflow only through MCP."""

    def __init__(self, session: Any) -> None:
        self.session = session

    async def run(self) -> None:
        """Run interactive CLI loop."""
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
        """Print menu."""
        print("\nInteractive inference workflow")
        self._print_settings(settings, compact=True)
        print("1. Process input")
        print("2. Change settings")
        print("3. Reset to defaults")
        print("4. View current settings")
        print("5. Exit")

    def _print_settings(self, settings: Dict[str, Any], compact: bool = False) -> None:
        """Print settings."""
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
        """Call MCP tool."""
        result = await self.session.call_tool(name, arguments)
        if getattr(result, "isError", False):
            messages: list[str] = []
            for block in getattr(result, "content", []):
                if getattr(block, "type", None) == "text":
                    messages.append(block.text)
                else:
                    messages.append(
                        json.dumps(block.model_dump(), indent=2, sort_keys=True)
                    )
            raise RuntimeError(
                "\n".join(messages) if messages else f"MCP tool failed: {name}"
            )

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
        """Get current settings from MCP server."""
        payload = await self._call_tool("get_inference_settings", {})
        if not isinstance(payload, dict):
            raise RuntimeError("MCP server returned invalid settings payload.")
        return payload

    async def view_settings(self, settings: Dict[str, Any] | None = None) -> None:
        """View current settings."""
        self._print_settings(settings or await self._get_settings(), compact=False)

    async def reset_defaults(self) -> None:
        """Reset to default settings."""
        await self._call_tool("reset_inference_settings", {})
        print("Defaults restored.")

    async def change_settings(self, settings: Dict[str, Any]) -> None:
        """Prompt user to change settings."""
        print("\nUpdate settings. Press Enter to keep the current value.")
        updates = {
            "input_mode": _prompt(
                "Input mode (single/batch)", settings["input_mode"]
            ).lower(),
            "task": _prompt(
                "Task (segmentation/classification/all)", settings["task"]
            ).lower(),
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
        """Process single or batch inputs."""
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
            print(
                f"Run finished. successes=1, failures=0, outputs={os.path.dirname(result['output_directory'])}"
            )
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
    workflow = MCPInferenceEngine(settings=settings)

    # Register signal handlers for graceful shutdown
    def _handle_shutdown_signal(signum: int, frame: Any) -> None:
        print("\nShutdown signal received. Stopping batch processing...")
        workflow.request_shutdown()

    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)

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
            return workflow.update_settings()

    def _process_case_impl(
        path: str,
        task: str = "all",
        qc_visualization: bool | None = None,
        output_dir: str | None = None,
        log_to_csv: bool | None = None,
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
        qc_visualization: bool | None = None,
        output_dir: str | None = None,
        log_to_csv: bool | None = None,
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
        qc_visualization: bool | None = None,
        output_dir: str | None = None,
        log_to_csv: bool | None = None,
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
        qc_visualization: bool | None = None,
        output_dir: str | None = None,
        log_to_csv: bool | None = None,
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
        qc_visualization: bool | None = None,
        output_dir: str | None = None,
        log_to_csv: bool | None = None,
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
        qc_visualization: bool | None = None,
        output_dir: str | None = None,
        log_to_csv: bool | None = None,
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
    """Run interactive CLI by connecting to MCP server."""
    if ClientSession is None or streamable_http_client is None:
        raise ImportError(
            'The MCP Python SDK is not installed. Install it with: pip install "mcp[cli]"'
        )

    if port <= 0:
        raise ValueError(
            "port must be a positive integer when starting the background MCP server"
        )

    server_log = tempfile.NamedTemporaryFile(
        mode="w+",
        encoding="utf-8",
        delete=False,
        prefix="inference_mcp_server_",
        suffix=".log",
    )
    server_log_path = server_log.name
    server_process: subprocess.Popen[str] | None = None
    normalized_mcp_path = _normalize_mcp_path(mcp_path)
    server_url = _build_mcp_server_url(
        host=host, port=port, mcp_path=normalized_mcp_path
    )
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


def _read_log_tail(log_path: str, max_chars: int = 4000) -> str:
    """Read tail of log file."""
    if not os.path.exists(log_path):
        return ""

    with open(log_path, encoding="utf-8", errors="replace") as handle:
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
    """Wait for background MCP server to start."""
    deadline = time.time() + timeout_seconds
    last_error: str | None = None

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
    """Shut down background MCP server gracefully."""
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
    """Build default session settings from config file."""
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
            key: list(value) for key, value in dict(classification_class_labels).items()
        },
        segmentation_num_classes=int(inference_cfg.get("segmentation_num_classes", 7)),
        preprocessing_resize_to=tuple(
            int(value)
            for value in inference_cfg.get("preprocessing_resize_to", [40, 224, 224])
        ),
        qc_visualization=bool(inference_cfg.get("qc_visualization", False)),
        csv_logging=bool(inference_cfg.get("csv_logging", True)),
        output_dir=_resolve_path(inference_cfg.get("output_dir"), cfg_base)
        or DEFAULT_OUTPUT_DIR,
        device=device if device != "auto" else cfg_device,
        inference_config_path=inference_config_path,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
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
    """Main entry point."""
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
