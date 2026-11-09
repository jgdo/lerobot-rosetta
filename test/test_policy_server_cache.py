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

"""RosettaPolicyServer preload/cache behavior.

The wrapper must load the policy once (at preload or the first handshake) and
reuse it for identical (policy_type, path, device) requests, while rebuilding
the cheap pre/post processors on every handshake. No real weights, no CUDA:
the policy factory and processor factory are monkeypatched.
"""

import pickle
from types import SimpleNamespace

import pytest
import torch
from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import RemotePolicyConfig

from lerobot_rosetta import policy_server as ps


class _FakePolicy:
    def __init__(self):
        self.config = SimpleNamespace()
        self.device = None

    def to(self, device):
        self.device = device
        return self


class _Recorder:
    """Counts from_pretrained and processor-factory calls via monkeypatch."""

    def __init__(self):
        self.loads: list[tuple[str, str]] = []
        self.processor_builds: list[dict] = []
        self.fail_paths: set[str] = set()

    def get_policy_class(self, policy_type):
        recorder = self

        class _Cls:
            @staticmethod
            def from_pretrained(path):
                if path in recorder.fail_paths:
                    raise OSError(f"no such repo: {path}")
                recorder.loads.append((policy_type, path))
                return _FakePolicy()

        return _Cls

    def make_pre_post_processors(self, *args, **kwargs):
        self.processor_builds.append(kwargs)
        return ("pre", "post")


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(ps, "get_policy_class", rec.get_policy_class)
    monkeypatch.setattr(ps, "make_pre_post_processors", rec.make_pre_post_processors)
    return rec


@pytest.fixture
def server():
    return ps.RosettaPolicyServer(PolicyServerConfig(host="127.0.0.1", port=8080))


def _handshake(server, **overrides):
    spec_kwargs = dict(
        policy_type="act",
        pretrained_name_or_path="user/repo",
        lerobot_features={},
        actions_per_chunk=10,
        device="cpu",
    )
    spec_kwargs.update(overrides)
    request = SimpleNamespace(data=pickle.dumps(RemotePolicyConfig(**spec_kwargs)))
    context = SimpleNamespace(peer=lambda: "test-client")
    server.Ready(SimpleNamespace(), context)  # what RobotClient.start() does first
    return server.SendPolicyInstructions(request, context)


def test_identical_handshakes_load_once(recorder, server):
    _handshake(server)
    _handshake(server)
    assert recorder.loads == [("act", "user/repo")]
    # Processors are rebuilt per handshake (rename_map comes from the client).
    assert len(recorder.processor_builds) == 2


def test_preload_makes_first_handshake_a_cache_hit(recorder, server):
    server.preload("act", "user/repo", "cpu")
    assert recorder.loads == [("act", "user/repo")]
    _handshake(server)
    assert recorder.loads == [("act", "user/repo")]  # no second load
    assert server.policy is not None
    assert len(recorder.processor_builds) == 1


def test_changed_path_or_device_reloads(recorder, server):
    _handshake(server)
    _handshake(server, pretrained_name_or_path="user/other")
    _handshake(server, pretrained_name_or_path="user/other", device="cuda")
    assert recorder.loads == [
        ("act", "user/repo"),
        ("act", "user/other"),
        ("act", "user/other"),
    ]


def test_failed_load_is_not_cached(recorder, server):
    recorder.fail_paths.add("user/repo")
    with pytest.raises(OSError):
        _handshake(server)
    assert server._loaded_key is None

    recorder.fail_paths.clear()
    _handshake(server)  # same key must actually load now, not hit a stale cache
    assert recorder.loads == [("act", "user/repo")]


def test_unsupported_policy_type_rejected(recorder, server):
    with pytest.raises(ValueError, match="not supported"):
        server.preload("not_a_policy", "user/repo", "cpu")
    assert recorder.loads == []


def _client_steps(server, chunk_len, rounds):
    """Drive RobotClient's real queue logic against the server's chunk timing.

    Each round: stamp an observation like control_loop_observation, time a
    chunk on the server, aggregate it into the client queue, then perform every
    queued action. Returns the performed timesteps.
    """
    import threading
    from queue import Queue

    from lerobot.async_inference.robot_client import RobotClient

    client = SimpleNamespace(
        action_queue=Queue(),
        action_queue_lock=threading.Lock(),
        latest_action=-1,
        latest_action_lock=threading.Lock(),
    )
    performed = []
    for _ in range(rounds):
        obs_timestep = max(client.latest_action, 0)
        chunk = server._time_action_chunk(0.0, [torch.zeros(1)] * chunk_len, obs_timestep)
        RobotClient._aggregate_action_queues(client, chunk)
        while not client.action_queue.empty():
            client.latest_action = client.action_queue.get_nowait().get_timestep()
            performed.append(client.latest_action)
    return performed


def test_single_action_chunks_keep_executing(server):
    # Upstream timing labels chunk[0] with the obs timestep, which the client
    # already performed, so every chunk after the first was dropped.
    assert _client_steps(server, chunk_len=1, rounds=5) == [1, 2, 3, 4, 5]


def test_multi_action_chunks_execute_every_action(server):
    assert _client_steps(server, chunk_len=3, rounds=2) == [1, 2, 3, 4, 5, 6]
