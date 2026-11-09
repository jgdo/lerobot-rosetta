#!/usr/bin/env python
# Copyright 2025 Isaac Blankenau
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LeRobot PolicyRunner.

Wraps LeRobot's async inference stack. It launches the gRPC policy-server
subprocess eagerly in setup() (node configure time) so the model is loaded
before the first goal arrives, then per goal drives a RobotClient whose control
loop pulls observations from and pushes actions to the injected frame I/O
(the node-hosted TopicBridge).
LeRobot-specific parameters are declared and read here so the hosting node
stays framework-agnostic.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import fields
from typing import Optional

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import RobotClient
from lerobot.utils.device_utils import is_torch_device_available
from rcl_interfaces.msg import ParameterDescriptor

# Submodule import, not the package root: pulling lerobot_robot_rosetta/__init__
# would import the full Robot stack just to get the config dataclass.
from lerobot_robot_rosetta.config_rosetta import RosettaConfig
from lerobot_robot_rosetta.rosetta import GATE_SLICE_SEC
from rosetta.contract.schema import Contract
from rosetta.contract.sidecar import resolve_repo_file
from rosetta.frames.protocols import FrameIO
from rosetta.policies import RunnerFeedback, RunnerResult

SERVER_STARTUP_TIMEOUT_SEC = 120.0  # default; covers model preload (see param)
SERVER_STARTUP_POLL_SEC = 0.5
SERVER_STOP_TIMEOUT_SEC = 5.0
THREAD_JOIN_TIMEOUT_SEC = 2.0
SERVER_LOG_TAIL_LINES = 50  # server output lines kept for error messages


def _resolve_policy_device(
    requested_device: str | None,
    is_available: Callable[[str], bool] = is_torch_device_available,
) -> str:
    """Resolve the requested policy device, falling back to CPU when its backend is unavailable.

    Availability is LeRobot's call (is_torch_device_available: cuda/mps/xpu/cpu), so this
    stays in sync with what the policy server will accept. Devices LeRobot doesn't know
    (it raises ValueError) pass through untouched — the server's own device checks are
    the authority there.
    """
    device = (requested_device or "").strip().lower()
    if not device:
        return "cpu"
    try:
        available = is_available(device.split(":", 1)[0])
    except ValueError:
        return device
    return device if available else "cpu"


def _read_checkpoint_policy_type(
    pretrained_name_or_path: str,
    warn: Callable[[str], None],
) -> str | None:
    """Return the "type" field of the checkpoint's config.json, or None.

    The checkpoint is a local directory or a Hub model id (resolve_repo_file
    handles both and reports Hub failures through warn).
    """
    config_path = resolve_repo_file(pretrained_name_or_path, "config.json", repo_type="model", warn=warn)
    if config_path is None:
        return None
    try:
        policy_type = json.loads(config_path.read_text()).get("type")
    except (OSError, ValueError, AttributeError) as e:
        warn(f"Could not read policy type from {config_path}: {e}")
        return None
    return policy_type if isinstance(policy_type, str) and policy_type else None


class LeRobotPolicyRunner:
    """PolicyRunner backed by LeRobot's RobotClient."""

    def __init__(self) -> None:
        self._node = None
        self._contract: Optional[Contract] = None
        self._contract_path: str = ""
        self._client: Optional[RobotClient] = None
        self._policy_device: str = "cpu"
        self._policy_type: str = ""
        self._server_process: Optional[subprocess.Popen] = None
        self._server_log_thread: Optional[threading.Thread] = None
        self._server_log_tail: deque[str] = deque(maxlen=SERVER_LOG_TAIL_LINES)

    # -------------------- Lifecycle --------------------

    def setup(self, node, contract: Contract) -> None:
        """Declare LeRobot inference parameters on node and store the contract."""
        self._node = node
        self._contract = contract
        # contract_path is declared by the node and shared by all backends. Read it.
        self._contract_path = node.get_parameter("contract_path").value

        # Fail the lifecycle configure early: the live LeRobot stack cannot
        # represent >1 numeric observation key or >1 action key (see
        # ensure_live_observation_layout).
        from lerobot_robot_rosetta.rosetta import ensure_live_observation_layout
        from rosetta.contract.specs import iter_action_specs, iter_observation_specs

        ensure_live_observation_layout(list(iter_observation_specs(contract)), list(iter_action_specs(contract)))

        declare = node.declare_parameter

        def _p(name, default, desc, read_only=False):
            if not node.has_parameter(name):
                declare(
                    name,
                    default,
                    ParameterDescriptor(description=desc, read_only=read_only),
                )

        _p("pretrained_name_or_path", "", "Path or HF repo ID of trained policy", True)
        _p("server_address", "127.0.0.1:8080", "Policy server address (host:port)", True)
        _p(
            "policy_type",
            "",
            "Policy architecture override (act, diffusion, ...). Empty = read 'type' from the checkpoint's config.json",
            True,
        )
        _p("policy_device", "cuda", "Device for policy inference (cuda, xpu, mps, cpu; cpu fallback)", True)
        _p("actions_per_chunk", 10, "Number of actions to request per chunk")
        _p("chunk_size_threshold", 0.5, "Queue threshold ratio to request new chunk (0.0-1.0)")
        _p("aggregate_fn_name", "weighted_average", "Action aggregation function")
        _p("launch_local_server", True, "Launch local policy server subprocess", True)
        _p("obs_similarity_atol", 1.0, "L2 tolerance for obs similarity (-1.0 disables)")
        _p("is_classifier", False, "Use reward section as action output", True)
        _p(
            "server_startup_timeout_sec",
            SERVER_STARTUP_TIMEOUT_SEC,
            "Max seconds to wait for the policy server to come up. Covers the "
            "model preload; raise it if the first launch must download weights.",
            True,
        )

        # Resolve once at configure time (the parameter is read-only): both the
        # client config and the server subprocess args use the resolved device.
        requested_device = node.get_parameter("policy_device").value
        self._policy_device = _resolve_policy_device(requested_device)
        if self._policy_device != requested_device:
            node.get_logger().warning(
                f"Requested policy device '{requested_device}' but the requested backend "
                f"is unavailable; using '{self._policy_device}' instead."
            )

        self._policy_type = self._resolve_policy_type()

        # Warm start: pay subprocess spawn + torch/CUDA init + model preload at
        # configure time so the first goal costs the same as any other. A
        # failure here propagates and fails the lifecycle configure transition.
        if node.get_parameter("launch_local_server").value:
            self._start_policy_server()

    def teardown(self) -> None:
        """Stop the policy server subprocess if running."""
        self._stop_policy_server()

    # -------------------- Execution --------------------

    def run(
        self,
        frames: FrameIO,
        *,
        task: str,
        stop_event: threading.Event,
    ) -> RunnerResult:
        """Run policy inference until completion or stop_event is set."""
        # Self-heal: the server is started eagerly in setup(); this respawns it
        # (with the same preload args — params are read-only) if it died.
        if self._node.get_parameter("launch_local_server").value:
            self._start_policy_server()

        if not self._policy_type:
            return RunnerResult(False, "No policy type: set pretrained_name_or_path (or policy_type)")

        robot_config = RosettaConfig(
            id="rosetta",
            config_path=self._contract_path,
            is_classifier=self._node.get_parameter("is_classifier").value,
        )
        robot_config._external_bridge = frames  # Inject the pre-built frame I/O
        # Under sim time the adapter's tick gates pace on the node clock; the
        # goal's stop_event releases a gate blocked by a paused sim clock.
        robot_config._stop_event = stop_event

        config = self._build_client_config(robot_config, task)
        client = RobotClient(config)
        self._client = client

        if not client.start():
            self._client = None
            return RunnerResult(False, "Failed to connect to policy server")

        receiver = threading.Thread(target=client.receive_actions, daemon=True)
        receiver.start()

        # Bridge the cooperative stop_event to LeRobot's shutdown_event.
        watcher_stop = threading.Event()
        watcher = threading.Thread(target=self._watch_stop, args=(stop_event, client, watcher_stop), daemon=True)
        watcher.start()

        try:
            client.control_loop(task=task)
        finally:
            client.stop()
            receiver.join(timeout=THREAD_JOIN_TIMEOUT_SEC)
            watcher_stop.set()
            watcher.join(timeout=THREAD_JOIN_TIMEOUT_SEC)

        cancelled = stop_event.is_set()
        self._client = None
        return RunnerResult(not cancelled, "Cancelled" if cancelled else "Completed")

    def request_stop(self) -> None:
        """Interrupt an in-progress control loop.

        Invariant: every stop path must set run()'s goal stop_event before (or
        with) this call. shutdown_event only exits the control loop at its
        while-check; under a paused sim the loop can be blocked inside a
        sim-paced robot call that only the stop_event releases.
        """
        # Local ref: run()'s finally can clear self._client from another thread.
        client = self._client
        if client is not None:
            client.shutdown_event.set()

    def feedback(self) -> RunnerFeedback:
        """Snapshot of action-queue depth and published-action count."""
        client = self._client
        if client is None:
            return RunnerFeedback(0, 0)
        with client.action_queue_lock:
            depth = client.action_queue.qsize()
        with client.latest_action_lock:
            published = max(0, client.latest_action)
        return RunnerFeedback(depth, published)

    # -------------------- Internals --------------------

    @staticmethod
    def _watch_stop(
        stop_event: threading.Event,
        client: RobotClient,
        watcher_stop: threading.Event,
    ) -> None:
        """Set the client's shutdown_event once the cooperative stop is requested.

        Polls at 10ms: once stop_event releases the adapter's sim-pacing gates,
        the control loop free-runs at the (lifted) wall fps until
        shutdown_event stops it, so this poll bounds that window.
        """
        while not watcher_stop.wait(0.01):
            if stop_event.is_set():
                client.shutdown_event.set()
                return

    def _resolve_policy_type(self) -> str:
        """Policy type from the checkpoint's config.json; the parameter overrides it.

        Raises when a checkpoint is set but its type can't be determined, so
        the lifecycle configure fails instead of the first goal.
        """
        node = self._node
        logger = node.get_logger()
        requested = node.get_parameter("policy_type").value.strip()
        pretrained = node.get_parameter("pretrained_name_or_path").value
        if not pretrained:
            return requested

        detected = _read_checkpoint_policy_type(pretrained, logger.warning)
        if requested:
            if detected and detected != requested:
                logger.warning(
                    f"policy_type '{requested}' overrides checkpoint config.json type '{detected}' ({pretrained})"
                )
            return requested
        if detected is None:
            raise RuntimeError(
                f"Could not determine the policy type from {pretrained}/config.json; "
                "set the policy_type parameter explicitly"
            )
        logger.info(f"Policy type '{detected}' read from checkpoint config.json ({pretrained})")
        return detected

    def _build_client_config(self, robot_config: RosettaConfig, task: str) -> RobotClientConfig:
        """Build RobotClientConfig from ROS2 parameters.

        Under sim time the robot adapter's tick gates pace the control loop on
        the node clock, so the client fps is only a wall-clock ceiling that
        must never bind before the gate itself runs out of resolution. The
        gate resolves at most one tick per GATE_SLICE_SEC, so that resolution
        rate IS the ceiling: faster gains nothing, slower throttles fast sims.
        On wall clock the contract fps paces the loop exactly as before.
        """
        node = self._node
        contract_fps = robot_config.fps
        # Auto-declared on every rclpy node by TimeSource.attach_node.
        use_sim_time = bool(node.get_parameter("use_sim_time").value)
        control_loop_fps = max(contract_fps, int(1 / GATE_SLICE_SEC)) if use_sim_time else contract_fps
        if use_sim_time:
            node.get_logger().info(
                f"Sim time active: pacing on the sim clock at contract fps={contract_fps}Hz; "
                f"client fps ceiling lifted to {control_loop_fps}Hz (FPSTracker logs the ceiling)."
            )

        config_kwargs = {
            "robot": robot_config,
            "server_address": node.get_parameter("server_address").value,
            "policy_type": self._policy_type,
            "pretrained_name_or_path": node.get_parameter("pretrained_name_or_path").value,
            "policy_device": self._policy_device,
            "task": task,
            "fps": control_loop_fps,
            "actions_per_chunk": node.get_parameter("actions_per_chunk").value,
            "chunk_size_threshold": node.get_parameter("chunk_size_threshold").value,
            "aggregate_fn_name": node.get_parameter("aggregate_fn_name").value,
        }

        # obs_similarity_atol is optional in some LeRobot versions. Gate on availability.
        atol_param = node.get_parameter("obs_similarity_atol").value
        atol = None if atol_param < 0 else atol_param
        supported = {f.name for f in fields(RobotClientConfig)}
        if "obs_similarity_atol" in supported:
            config_kwargs["obs_similarity_atol"] = atol
        elif atol_param != 1.0:
            node.get_logger().warning("obs_similarity_atol not supported in this LeRobot version; ignoring.")

        return RobotClientConfig(**config_kwargs)

    def _start_policy_server(self) -> None:
        """Launch the local policy server subprocess (idempotent while running)."""
        if self._server_process is not None and self._server_process.poll() is None:
            return  # already running

        node = self._node
        server_address = node.get_parameter("server_address").value
        host, port = server_address.split(":")

        if node.get_parameter("is_classifier").value:
            module = "lerobot_rosetta.classifier_server"
        else:
            module = "lerobot_rosetta.policy_server"

        cmd = [sys.executable, "-m", module, f"--host={host}", f"--port={port}"]
        pretrained = node.get_parameter("pretrained_name_or_path").value
        if pretrained:
            cmd += [
                f"--policy-type={self._policy_type}",
                f"--pretrained-name-or-path={pretrained}",
                f"--policy-device={self._policy_device}",
            ]
        else:
            node.get_logger().warning(
                "pretrained_name_or_path is empty; the policy server will load "
                "the model on the first goal instead of at startup"
            )
        timeout = node.get_parameter("server_startup_timeout_sec").value

        node.get_logger().info(
            f"Launching {module} on {host}:{port} (preload={pretrained or 'none'}, timeout={timeout}s)..."
        )
        # Pipe stdout+stderr through a reader thread that echoes to the console
        # and keeps a tail for error messages (it also drains the pipe so the
        # server can never block on a full buffer).
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self._server_log_tail.clear()
        self._server_process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self._server_log_thread = threading.Thread(
            target=self._pump_server_output, args=(self._server_process.stdout,), daemon=True
        )
        self._server_log_thread.start()

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._server_process.poll() is not None:
                raise RuntimeError(
                    f"Policy server exited with code {self._server_process.returncode}. "
                    f"Last output:\n{self._read_log_tail()}"
                )
            try:
                with socket.create_connection((host, int(port)), timeout=1.0):
                    node.get_logger().info(f"Policy server ready on {host}:{port}")
                    return
            except (ConnectionRefusedError, socket.timeout, OSError):
                time.sleep(SERVER_STARTUP_POLL_SEC)

        raise RuntimeError(f"Policy server failed to start within {timeout}s. Last output:\n{self._read_log_tail()}")

    def _pump_server_output(self, stream) -> None:
        """Echo server output to the console and remember the last lines."""
        for raw in iter(stream.readline, b""):
            line = raw.decode("utf-8", "replace")
            self._server_log_tail.append(line)
            sys.stdout.write(f"[policy_server] {line}")
            sys.stdout.flush()
        stream.close()

    def _read_log_tail(self) -> str:
        """Last lines of server output, for diagnostics."""
        proc = self._server_process
        if self._server_log_thread is not None and proc is not None and proc.poll() is not None:
            # Let the reader catch up with output written just before exit.
            self._server_log_thread.join(timeout=THREAD_JOIN_TIMEOUT_SEC)
        return "".join(self._server_log_tail) or "(no server output)"

    def _stop_policy_server(self) -> None:
        """Terminate the policy server process if running."""
        proc = self._server_process
        if proc is None or proc.poll() is not None:
            self._join_server_log_thread()
            self._server_process = None
            return
        if self._node is not None:
            self._node.get_logger().info("Stopping local policy server...")
        proc.terminate()
        try:
            proc.wait(timeout=SERVER_STOP_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=SERVER_STOP_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                if self._node is not None:
                    self._node.get_logger().error("Policy server did not exit after kill().")
        self._join_server_log_thread()
        self._server_process = None

    def _join_server_log_thread(self) -> None:
        if self._server_log_thread is not None:
            self._server_log_thread.join(timeout=THREAD_JOIN_TIMEOUT_SEC)
            self._server_log_thread = None
