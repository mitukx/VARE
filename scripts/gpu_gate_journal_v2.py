"""CPU-testable checkpoint, timeout, and token-budget logic for GPU gates."""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path
from typing import Callable, Iterable


class GateWallTimeExceeded(TimeoutError):
    pass


def completion_token_count(token_ids: Iterable[int], eos_token_id: int | None) -> int:
    """Count generated IDs, including the first EOS and excluding later padding."""
    count = 0
    for token_id in token_ids:
        count += 1
        if eos_token_id is not None and int(token_id) == eos_token_id:
            break
    return count


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class GateJournal:
    """Append-only prompt records with identity-checked resume behavior."""

    def __init__(self, output_path: Path, identity: dict, *, resume: bool = False):
        self.output_path = output_path
        self.partial_path = output_path.with_suffix(output_path.suffix + ".partial.jsonl")
        self.state_path = output_path.with_suffix(output_path.suffix + ".state.json")
        self.identity = identity
        self.records: dict[int, dict] = {}
        self.previous_elapsed = 0.0
        self.started = time.monotonic()
        paths_exist = any(path.exists() for path in (self.output_path, self.partial_path, self.state_path))
        if resume:
            if not self.state_path.is_file():
                raise ValueError("cannot resume: state file is missing")
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("identity") != identity:
                raise ValueError("cannot resume: model/data/protocol identity differs")
            if state.get("decision") == "complete":
                raise ValueError("cannot resume: gate is already complete")
            self.previous_elapsed = float(state.get("elapsed_seconds", 0.0))
            if self.partial_path.exists():
                for line_no, line in enumerate(self.partial_path.read_text(encoding="utf-8").splitlines(), 1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    index = record.get("prompt_index")
                    if not isinstance(index, int) or index in self.records:
                        raise ValueError(f"invalid or duplicated partial prompt index at line {line_no}")
                    self.records[index] = record
        elif paths_exist:
            raise FileExistsError("gate output exists; use --resume or choose a new path")
        else:
            self.write_state("running", None)

    def elapsed(self) -> float:
        return self.previous_elapsed + (time.monotonic() - self.started)

    def has(self, prompt_index: int) -> bool:
        return prompt_index in self.records

    def append(self, record: dict) -> None:
        index = record.get("prompt_index")
        if not isinstance(index, int) or index in self.records:
            raise ValueError("prompt index must be a new integer")
        self.partial_path.parent.mkdir(parents=True, exist_ok=True)
        with self.partial_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.records[index] = record
        self.write_state("running", None)

    def write_state(self, decision: str, reason: str | None) -> None:
        _atomic_json(
            self.state_path,
            {
                "identity": self.identity,
                "decision": decision,
                "reason": reason,
                "elapsed_seconds": self.elapsed(),
                "completed_prompt_groups": len(self.records),
                "partial_path": self.partial_path.name,
            },
        )

    def stop(self, decision: str, reason: str) -> None:
        self.write_state(decision, reason)

    def finalize(self, result: dict) -> None:
        result["prompt_groups"] = [self.records[index] for index in sorted(self.records)]
        result["elapsed_seconds"] = self.elapsed()
        _atomic_json(self.output_path, result)
        self.write_state("complete", None)


def wall_time_limit(seconds: float):
    """Raise inside Python on POSIX; batch checkpoints remain durable on expiry."""
    if seconds <= 0:
        raise ValueError("max_wall_seconds must be positive")
    if not hasattr(signal, "setitimer") or not hasattr(signal, "SIGALRM"):
        return _NoopTimer()
    return _SignalTimer(seconds)


class _NoopTimer:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _SignalTimer:
    def __init__(self, seconds: int):
        self.seconds = seconds

    def __enter__(self):
        self.old_handler = signal.getsignal(signal.SIGALRM)
        self.old_timer = signal.getitimer(signal.ITIMER_REAL)
        signal.signal(signal.SIGALRM, self._raise)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    @staticmethod
    def _raise(_signum, _frame):
        raise GateWallTimeExceeded("wall-time limit reached")

    def __exit__(self, *_exc):
        signal.setitimer(signal.ITIMER_REAL, *self.old_timer)
        signal.signal(signal.SIGALRM, self.old_handler)
        return False


def run_prompt_groups(
    *,
    rows: list[dict],
    journal: GateJournal,
    generate: Callable[[int, dict, int], dict],
    completions_per_prompt: int,
    max_new_tokens_per_completion: int,
    max_generated_tokens: int,
    max_wall_seconds: int,
    eos_token_id: int | None,
) -> str:
    """Run/resume one prompt group at a time; always retain completed groups."""
    if completions_per_prompt <= 0 or max_new_tokens_per_completion <= 0:
        raise ValueError("completion counts and per-completion token limit must be positive")
    if max_generated_tokens <= 0:
        raise ValueError("max_generated_tokens must be positive")
    generated_before = sum(int(r["generated_tokens"]) for r in journal.records.values())
    remaining_wall = max_wall_seconds - journal.elapsed()
    if remaining_wall <= 0:
        journal.stop("wall_time", "wall-time budget was exhausted before this attempt")
        return "wall_time"
    try:
        with wall_time_limit(remaining_wall):
            for prompt_index, row in enumerate(rows):
                if journal.has(prompt_index):
                    continue
                remaining = max_generated_tokens - generated_before
                per_completion_cap = min(max_new_tokens_per_completion, remaining // completions_per_prompt)
                if per_completion_cap < 1:
                    journal.stop("token_budget", "insufficient budget for one token per completion")
                    return "token_budget"
                group = generate(prompt_index, row, per_completion_cap)
                token_ids = group.pop("completion_token_ids")
                if len(token_ids) != completions_per_prompt:
                    raise ValueError("generator returned an incomplete completion group")
                lengths = [completion_token_count(ids, eos_token_id) for ids in token_ids]
                if any(length > per_completion_cap for length in lengths):
                    raise ValueError("generator returned more token IDs than requested")
                group.update(
                    {
                        "prompt_index": prompt_index,
                        "completion_token_ids": token_ids,
                        "completion_token_counts": lengths,
                        "generated_tokens": sum(lengths),
                        "max_new_tokens_per_completion": per_completion_cap,
                    }
                )
                journal.append(group)
                generated_before += sum(lengths)
                if journal.elapsed() >= max_wall_seconds:
                    journal.stop("wall_time", "wall-time limit reached after a completed prompt group")
                    return "wall_time"
    except GateWallTimeExceeded as error:
        journal.stop("wall_time", str(error))
        return "wall_time"
    except BaseException as error:
        name = type(error).__name__
        decision = "oom" if "outofmemory" in name.casefold() else "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        journal.stop(decision, f"{name}: {error}")
        raise
    return "complete"
