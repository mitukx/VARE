"""Observe whether failed evaluation leaves a sibling using a discarded state."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
sys.path.insert(0, str(args.source / "src"))

from vare.config import EngineConfig
from vare.engine import CapabilityLoop
from vare.types import Attempt, EvaluationReport, Task, Verification
from vare.verifiers import VerifierEnsemble, VerifierMember


class Verifier:
    name = "fixture"
    version = 0
    trusted = True

    async def verify(self, attempt):
        return Verification(1.0, True, 1.0, 0, self.name, trusted=True)


class Hooks:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()
        self.states = {"p0": 0}
        self.events = []
        self.tasks = []
        self.candidate_error = None

    def active_policy(self):
        return "p0", 0

    async def rollout(self, task, policy_id, policy_version, step):
        return Attempt(task, "ok", policy_id, policy_version, step)

    async def train_candidate(self, incumbent_id, experiences):
        self.states["p1"] = 1
        return "p1"

    async def evaluate(self, policy_id):
        self.tasks.append(asyncio.current_task())
        if policy_id == "p0":
            await self.started.wait()
            raise RuntimeError("incumbent evaluator failure")
        self.events.append("candidate-started")
        self.started.set()
        try:
            await self.release.wait()
            return EvaluationReport(policy_id, self.states[policy_id], n=1)
        except BaseException as exc:
            self.candidate_error = type(exc).__name__
            raise
        finally:
            self.events.append("candidate-finished")
            self.finished.set()

    async def promote(self, candidate_id):
        raise AssertionError("must not promote")

    async def discard(self, candidate_id):
        self.events.append("discard")
        del self.states[candidate_id]


async def main():
    hooks = Hooks()
    loop = CapabilityLoop(hooks=hooks, verifier=VerifierEnsemble([VerifierMember(Verifier())]),
                          config=EngineConfig(replay_batch_size=1))
    error = None
    try:
        await loop.run_round([Task("t", "prompt")], round_index=0)
    except RuntimeError as exc:
        error = str(exc)
    result = {"round_error": error, "evaluation_live_after_round_return": not hooks.finished.is_set(),
              "round_lock_held_after_return": loop._round_lock.locked(),
              "candidate_snapshot_exists_after_return": "p1" in hooks.states,
              "events_at_round_return": list(hooks.events)}
    hooks.release.set()
    await asyncio.gather(*hooks.tasks, return_exceptions=True)
    result["candidate_error"] = hooks.candidate_error
    result["final_events"] = hooks.events
    result["discard_after_evaluation_finish"] = hooks.events.index("candidate-finished") < hooks.events.index("discard")
    args.output.write_text(json.dumps(result, indent=2) + "\n")


asyncio.run(main())
