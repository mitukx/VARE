#!/usr/bin/env python3
"""Replay the exact pinned TRL environment-column method with a bounded oracle."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/trl_8520_environment_column_source_v1.json"


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def read_exact(url, *, expected, limit, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        data = response.read(limit + 1)
    if len(data) > limit or git_blob(data) != expected:
        raise ValueError("upstream source size or Git blob mismatch")
    return data


def method_from_source(data, class_name, method_name):
    tree = ast.parse(data)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
    if len(classes) != 1:
        raise ValueError("expected exactly one target class")
    methods = [
        node for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    ]
    if len(methods) != 1:
        raise ValueError("expected exactly one target method")
    return methods[0]


def assignment_columns(node):
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        raise ValueError("only one signature assignment is allowed")
    target = node.targets[0]
    if (not isinstance(target, ast.Attribute) or target.attr != "_signature_columns"
            or not isinstance(target.value, ast.Name) or target.value.id != "self"):
        raise ValueError("assignment target outside signature field")
    if not isinstance(node.value, (ast.List, ast.Tuple)):
        raise ValueError("expected literal column names")
    if not all(isinstance(x, ast.Constant) and type(x.value) is str for x in node.value.elts):
        raise ValueError("column names must be strings")
    return [x.value for x in node.value.elts]


def checked_method(fn, *, allow_unconditional=False):
    # This source-only harness never evaluates the surrounding class/module.
    # Restrict the function to the original simple structure (or one named
    # deliberately wrong unconditional-assignment control).
    args = fn.args
    if (fn.decorator_list or fn.returns is not None or fn.type_comment is not None
            or fn.name != "_set_signature_columns_if_needed"
            or len(args.args) != 1 or args.args[0].arg != "self"
            or args.args[0].annotation is not None or args.posonlyargs
            or args.kwonlyargs or args.defaults or args.kw_defaults
            or args.vararg or args.kwarg or getattr(fn, "type_params", [])):
        raise ValueError("unexpected signature")
    if len(fn.body) != 1:
        raise ValueError("unexpected method body")
    body = fn.body[0]
    if isinstance(body, ast.If):
        expected = ast.parse("self._signature_columns is None", mode="eval").body
        if ast.dump(body.test) != ast.dump(expected) or len(body.body) != 1 or body.orelse:
            raise ValueError("unexpected signature initialization guard")
        assignment_columns(body.body[0])
    elif allow_unconditional:
        assignment_columns(body)
    else:
        raise ValueError("expected guarded signature initialization")
    namespace = {"__builtins__": {}}
    code = ast.Module(body=[copy.deepcopy(fn)], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), "<pinned-signature-method>", "exec"), namespace)
    return namespace[fn.name]


def mutant(fn, name):
    modified = copy.deepcopy(fn)
    guard = modified.body[0]
    assign = guard.body[0]
    columns = assignment_columns(assign)
    if name == "wrong_plural":
        columns = ["environments" if x == "environment" else x for x in columns]
    elif name == "drop_image":
        columns = [x for x in columns if x != "image"]
    elif name == "overwrite_existing":
        modified.body = [assign]
    else:
        raise ValueError("unknown mutant")
    assign.value = ast.List(elts=[ast.Constant(value=x) for x in columns], ctx=ast.Load())
    return ast.fix_missing_locations(modified)


def exercise(method):
    class Stub:
        def __init__(self, columns=None):
            self._signature_columns = columns

    def projected(row):
        trainer = Stub()
        method(trainer)
        return {key: value for key, value in row.items() if key in trainer._signature_columns}

    # The oracle tests the externally relevant projection and dispatch
    # contract, not equality with the known fixed implementation.
    environment_only = projected({"environment": "counter", "unused": 17})
    mixed = [
        projected({"prompt": "a", "environment": "counter", "unused": 1}),
        projected({"prompt": "b", "environment": "echo", "unused": 2}),
    ]
    factories = {"counter": "counter-instance", "echo": "echo-instance"}
    routes = [factories.get(row.get("environment")) for row in mixed]
    images = projected({"prompt": "p", "image": "one", "images": ["two"], "discard": 3})
    existing = ["custom_signature"]
    trainer = Stub(existing)
    method(trainer)
    return {
        "environment_only": environment_only == {"environment": "counter"},
        "mixed_environment_selection": (
            routes == ["counter-instance", "echo-instance"]
            and [row.get("prompt") for row in mixed] == ["a", "b"]
            and all("unused" not in row for row in mixed)
        ),
        "preserve_prompt_and_image_columns": images == {"prompt": "p", "image": "one", "images": ["two"]},
        "preserve_existing_signature": trainer._signature_columns is existing,
    }


def run(output):
    if output.exists():
        raise FileExistsError(output)
    p = json.loads(PROTOCOL.read_text())
    source, limits = p["source"], p["limits"]
    root = f"https://raw.githubusercontent.com/{source['repository']}/"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    record = {
        "schema": "vare.source_qualification.trl_environment_column_result/v1",
        "source_commit": head, "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "allocation": "UNASSIGNED_SOURCE_ONLY", "status": "INCOMPLETE",
        "model_inference": False, "optimizer_updated": False, "heldout_accessed": False,
        "upstream_training_test_executed": False, "source_files": [], "arms": [],
    }

    def fetch(commit, path, blob, limit):
        raw = read_exact(root + commit + "/" + path, expected=blob,
                         limit=limit, timeout=limits["download_timeout_seconds"])
        record["source_files"].append({"commit": commit, "path": path, "git_blob": git_blob(raw),
                                       "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
        return raw

    try:
        license_bytes = fetch(source["fixed_commit"], source["license_path"],
                              source["license_blob"], limits["max_license_bytes"])
        if b"Apache License" not in license_bytes or b"Version 2.0" not in license_bytes:
            raise ValueError("unexpected license")
        upstream_test = fetch(source["fixed_commit"], source["test_path"],
                              source["test_blob"], limits["max_test_bytes"])
        if source["upstream_test"].encode() not in upstream_test:
            raise ValueError("upstream regression selector absent")
        functions = {}
        for arm in ("baseline", "fixed"):
            raw = fetch(source[f"{arm}_commit"], source["path"],
                        source[f"{arm}_blob"], limits["max_source_bytes"])
            functions[arm] = method_from_source(raw, p["methods"]["class"], p["methods"]["name"])
        for arm in p["arms"]:
            fn = functions[arm] if arm in functions else mutant(functions["fixed"], arm)
            case_results = exercise(checked_method(fn, allow_unconditional=arm == "overwrite_existing"))
            if sorted(case_results) != sorted(p["cases"]):
                raise ValueError("oracle case inventory mismatch")
            row = {"arm": arm, "cases": case_results, "pass": all(case_results.values()),
                   "ast_sha256": hashlib.sha256(ast.dump(fn).encode()).hexdigest()}
            record["arms"].append(row)
            print("VARE_TRL8520_ARM " + json.dumps(row, sort_keys=True), flush=True)
        actual = {row["arm"]: row["pass"] for row in record["arms"]}
        record["status"] = "PASS_SOURCE_REPLAY" if actual == p["expected"] else "NO_GO_SOURCE_REPLAY"
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        with output.open("x") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print("VARE_TRL8520_FINAL " + json.dumps({
            "status": record["status"], "source_commit": head, "protocol_sha256": record["protocol_sha256"],
            "result_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "error": record.get("error"),
        }, sort_keys=True), flush=True)
    if record["status"] != "PASS_SOURCE_REPLAY":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
