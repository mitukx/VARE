"""Calibrate two fresh Development module-repair contracts using existing isolation.

This is an operator-owned experiment, not an arbitrary public grading service.
The committed protocol hash must be supplied from outside the working tree.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import FunctionType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vare import isolated_candidate as grading  # noqa: E402
from vare.container_isolation import DockerIsolationRunner, IsolationLimits  # noqa: E402


def sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def null_safe_grade_calls(*args, **kwargs):
    """Reuse the real host grader with private recursive null comparison.

    Follows the corrected #102 v10 private-globals approach; frozen shared
    runtime bytes and historical protocols are deliberately unchanged.
    """
    original = grading.values_close
    namespace = original.__globals__.copy()
    numeric = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)
    numeric.__kwdefaults__ = original.__kwdefaults__

    def compare(actual, expected, **options):
        if actual is None or expected is None:
            return actual is None and expected is None
        return numeric(actual, expected, **options)

    namespace["values_close"] = compare
    parent = grading.grade_calls
    parent_globals = parent.__globals__.copy()
    parent_globals["values_close"] = compare
    wrapped = FunctionType(parent.__code__, parent_globals, parent.__name__, parent.__defaults__)
    wrapped.__kwdefaults__ = parent.__kwdefaults__
    return wrapped(*args, **kwargs)


def source_bytes(repository: Path, revision: str, path: str) -> bytes:
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("source revision must be a full immutable SHA")
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_NO_REPLACE_OBJECTS="1", GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    return subprocess.run(
        ["git", "--no-replace-objects", "-c", "core.hooksPath=/dev/null", "-c",
         "core.fsmonitor=false", "-C", str(repository), "show", f"{revision}:{path}"],
        env=env, check=True, capture_output=True, timeout=30,
    ).stdout


def load_protocol(path: Path, expected_sha: str) -> dict[str, Any]:
    content = path.read_bytes()
    if sha(content) != expected_sha:
        raise ValueError("operator-supplied protocol identity mismatch")
    protocol = json.loads(content)
    for name, expected in protocol["implementation_sha256"].items():
        if sha((ROOT / name).read_bytes()) != expected:
            raise ValueError(f"implementation drift: {name}")
    return protocol


def thinking_calls():
    text = "{{ messages[0]['content'] }}"
    cases = []

    def add(label, tokens, template, markers, encoded, rendered="hi", kwargs=None):
        cases.append(grading.FunctionCall(label, "adapter", "run", [{
            "tokens": tokens, "template": template, "kwargs": kwargs or {},
        }], {"start": markers[0], "end": markers[1],
             "start_tokens": encoded[0], "end_tokens": encoded[1], "rendered": rendered}))

    for label, pair in [("bracket", ["[THINK]", "[/THINK]"]),
                        ("existing-angle", ["<think>", "</think>"]),
                        ("existing-longcat", ["<longcat_think>", "</longcat_think>"])]:
        add(label, pair, text, pair, [[1], [2]])
    for label, template in [("text-marker", "{{ '<think>' }}"),
                             ("dict-template", {"default": "{{ '<think>' }}"})]:
        add(label, [], template, ["<think>", "</think>"], [[0], [0]], "<think>")
    add("ordinary-null", [], text, [None, None], [None, None])
    add("one-marker-only", ["[THINK]"], text, [None, None], [None, None])
    template = "{% if thinking is defined and thinking %}<think>{% endif %}"
    add("thinking-enabled", ["<think>", "</think>"], template,
        ["<think>", "</think>"], [[1], [2]], "<think>")
    add("thinking-disabled", ["<think>", "</think>"], template,
        ["<think>", "</think>"], [[1], [2]], "", {"enable_thinking": False})
    return cases


def trusted_reference(test_source: bytes, image: str, case: dict, output: Path):
    """Run the pinned independent immediate-update oracle without any candidate."""
    tree = ast.parse(test_source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "reference_gptq")
    reference = ast.get_source_segment(test_source.decode(), node)
    code = "import json\nimport mlx.core as mx\n" + reference + "\n" + f"""
mx.random.seed({case['seed']})
weight = mx.random.normal((4, {case['features']}))
data = mx.random.normal((128, {case['features']}))
data = data + 0.5 * mx.sum(data, axis=-1, keepdims=True)
mx.eval(weight, data)
expected = reference_gptq(weight, data, {case['bits']}, 32)
mx.eval(expected)
print(json.dumps({{'request': {{'weight': weight.tolist(), 'data': data.tolist(),
    'bits': {case['bits']}, 'group_size': 32}}, 'expected': expected.tolist()}}, allow_nan=False))
"""
    runner = DockerIsolationRunner()
    limits = IsolationLimits(image=image, timeout_s=45, memory_mb=512, cpus=1, pids_limit=64,
                             max_output_bytes=512000)
    command = runner._common(limits, "vare-reference-" + os.urandom(8).hex())
    started = time.monotonic()
    run = subprocess.run([*command, image, "python", "-c", code], env=runner._client_env,
                         capture_output=True, timeout=50)
    output.write_text(json.dumps({"returncode": run.returncode, "stdout": run.stdout.decode(),
                                 "stderr": run.stderr.decode(), "elapsed_seconds": time.monotonic()-started,
                                 "code_sha256": sha(code.encode())}, indent=2) + "\n")
    run.check_returncode()
    return json.loads(run.stdout)


def grade_source(content: bytes, adapter: bytes, calls, protocol):
    with tempfile.TemporaryDirectory(prefix="vare-flagship-") as directory:
        subject = Path(directory) / "subject"
        subject.mkdir()
        (subject / "target.py").write_bytes(content)
        (subject / "adapter.py").write_bytes(adapter)
        return null_safe_grade_calls(
            subject, calls, task_kind="correctness", image=protocol["image"],
            timeout_s=45, memory_mb=512, rtol=1e-5, atol=1e-5,
            discard_candidate_stdout=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repository", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol, args.protocol_sha256)
    args.output.mkdir(parents=True, exist_ok=False)
    image_info = subprocess.run(["docker", "--context", "colima", "image", "inspect", protocol["image"]],
                                check=True, capture_output=True, timeout=10)
    (args.output / "image-inspect.json").write_bytes(image_info.stdout)
    outcomes = []
    for task in protocol["tasks"]:
        name = task["id"]
        directory = args.output / name
        directory.mkdir()
        baseline = source_bytes(args.source_repository, task["baseline_commit"], task["target"])
        fixed = source_bytes(args.source_repository, task["fixed_commit"], task["target"])
        test = source_bytes(args.source_repository, task["fixed_commit"], task["upstream_test_path"])
        for key, value in [("baseline_sha256", baseline), ("fixed_sha256", fixed),
                           ("upstream_test_sha256", test)]:
            if sha(value) != task[key]:
                raise ValueError(f"immutable upstream source mismatch: {name}/{key}")
        adapter = (ROOT / task["adapter"]).read_bytes()
        if name == "thinking-markers":
            calls = thinking_calls()
        else:
            calls = []
            for item in protocol["gptq_cases"]:
                oracle = trusted_reference(test, protocol["image"], item, directory / (item["id"] + "-reference.json"))
                calls.append(grading.FunctionCall(item["id"], "adapter", "run", [oracle["request"]], oracle["expected"]))
        # Mutations are fixed in the protocol before any calibration outcome.
        variants = {"baseline": baseline, "known-fix": fixed}
        for mutation in task["mutations"]:
            before, after = mutation["old"].encode(), mutation["new"].encode()
            if fixed.count(before) != 1:
                raise ValueError("mutation must select one unambiguous source span")
            variants[mutation["id"]] = fixed.replace(before, after, 1)
        for variant, content in variants.items():
            started = time.monotonic()
            report = grade_source(content, adapter, calls, protocol)
            record = {"task": name, "variant": variant, "source_sha256": sha(content),
                      "elapsed_seconds": time.monotonic()-started, **asdict(report)}
            (directory / (variant + ".json")).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            outcomes.append(record)
            print(json.dumps({"task": name, "variant": variant, "passed": report.passed,
                              "cases_passed": sum(c.passed for c in report.cases), "cases": len(report.cases)}), flush=True)
    summary = {"protocol_sha256": args.protocol_sha256, "outcomes": outcomes,
               "qualification_passed": all(r["passed"] == (r["variant"] == "known-fix") for r in outcomes),
               "model_inference": "NOT_RUN", "optimizer_steps": 0}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0 if summary["qualification_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
