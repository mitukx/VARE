#!/usr/bin/env python3
"""Run isolated reference and incorrect-mutant controls for VARE-IG Train tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts import qwen3_function_repair_grpo as runner  # noqa: E402
from vare.learning_pipeline import vare_ig_tasks as tasks  # noqa: E402


REFERENCES = {
    "vare_ig.train.ini_sections.v1": '''def solve(lines):
    result = {}
    section = ""
    for line in lines:
        text = line.strip()
        if not text or text.startswith("#") or text.startswith(";"):
            continue
        if text.startswith("[") and text.endswith("]") and text[1:-1].strip():
            section = text[1:-1].strip()
            result.setdefault(section, {})
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if key:
                result.setdefault(section, {})[key] = value
    return result
''',
    "vare_ig.train.csv_record.v1": '''def solve(text):
    if text == "":
        return [""]
    fields = []
    i = 0
    while True:
        field = []
        if i < len(text) and text[i] == '"':
            i += 1
            closed = False
            while i < len(text):
                if text[i] == '"':
                    if i + 1 < len(text) and text[i + 1] == '"':
                        field.append('"')
                        i += 2
                    else:
                        i += 1
                        closed = True
                        break
                else:
                    field.append(text[i])
                    i += 1
            if not closed or (i < len(text) and text[i] != ","):
                return None
        else:
            while i < len(text) and text[i] != ",":
                if text[i] == '"':
                    return None
                field.append(text[i])
                i += 1
        fields.append("".join(field))
        if i == len(text):
            return fields
        i += 1
        if i == len(text):
            fields.append("")
            return fields
''',
    "vare_ig.train.cli_overrides.v1": '''def solve(tokens):
    result = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            i += 1
            continue
        body = token[2:]
        if "=" in body:
            key, value = body.split("=", 1)
            if key:
                result[key.replace("-", "_")] = value
            i += 1
        elif body.startswith("no-"):
            key = body[3:]
            if key:
                result[key.replace("-", "_")] = False
            i += 1
        else:
            key = body.replace("-", "_")
            if not key:
                i += 1
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                result[key] = tokens[i + 1]
                i += 2
            else:
                result[key] = "true"
                i += 1
    return result
''',
    "vare_ig.train.stable_toposort.v1": '''def solve(nodes, edges):
    nodes = sorted(set(nodes))
    node_set = set(nodes)
    if any(a not in node_set or b not in node_set for a, b in edges):
        return None
    outgoing = {node: set() for node in nodes}
    indegree = {node: 0 for node in nodes}
    for a, b in edges:
        if b not in outgoing[a]:
            outgoing[a].add(b)
            indegree[b] += 1
    ready = sorted(node for node in nodes if indegree[node] == 0)
    result = []
    while ready:
        node = ready.pop(0)
        result.append(node)
        for neighbor in sorted(outgoing[node]):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                ready.append(neighbor)
                ready.sort()
    return result if len(result) == len(nodes) else None
''',
    "vare_ig.train.graph_reachable.v1": '''def solve(graph, start):
    nodes = set(graph)
    for neighbors in graph.values():
        nodes.update(neighbors)
    if start not in nodes:
        return []
    reached = {start}
    pending = [start]
    while pending:
        node = pending.pop()
        for neighbor in graph.get(node, []):
            if neighbor not in reached:
                reached.add(neighbor)
                pending.append(neighbor)
    return sorted(reached)
''',
    "vare_ig.train.undirected_components.v1": '''def solve(graph):
    adjacency = {node: set() for node in graph}
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            adjacency.setdefault(neighbor, set())
            adjacency[node].add(neighbor)
            adjacency[neighbor].add(node)
    unseen = set(adjacency)
    components = []
    while unseen:
        start = min(unseen)
        pending = [start]
        unseen.remove(start)
        component = []
        while pending:
            node = pending.pop()
            component.append(node)
            for neighbor in sorted(adjacency[node]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    pending.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda component: component[0])
''',
    "vare_ig.train.merge_busy_windows.v1": '''def solve(windows, buffer):
    if buffer < 0:
        return None
    result = []
    for start, end in sorted(windows):
        if result and start <= result[-1][1] + buffer:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result
''',
    "vare_ig.train.running_key_totals.v1": '''def solve(events):
    totals = {}
    result = []
    for key, amount in events:
        totals[key] = totals.get(key, 0) + amount
        result.append([key, totals[key]])
    return result
''',
    "vare_ig.train.latest_record_by_key.v1": '''def solve(records):
    latest = {}
    for index, record in enumerate(records):
        key, value, timestamp = record
        if key not in latest or timestamp >= latest[key][0]:
            latest[key] = (timestamp, index, record)
    return [latest[key][2] for key in sorted(latest)]
''',
    "vare_ig.train.deep_merge_defaults.v1": '''def solve(config, defaults):
    result = {}
    for key, value in config.items():
        if key in defaults and type(value) is dict and type(defaults[key]) is dict:
            result[key] = solve(value, defaults[key])
        else:
            result[key] = value
    for key, value in defaults.items():
        if key not in result:
            result[key] = value.copy() if type(value) is dict else value
    return result
''',
    "vare_ig.train.normalize_relative_path.v1": '''def solve(path):
    if path.startswith("/"):
        return None
    parts = []
    for part in path.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                return None
            parts.pop()
        else:
            parts.append(part)
    return "/".join(parts)
''',
    "vare_ig.train.parse_duration_ms.v1": '''def solve(text):
    text = text.strip()
    unit = None
    for suffix, multiplier in (("ms", 1), ("s", 1000), ("m", 60000), ("h", 3600000)):
        if text.endswith(suffix):
            unit = multiplier
            number = text[:-len(suffix)].strip()
            break
    if unit is None or not number or number.count(".") > 1:
        return None
    digits = number.replace(".", "")
    if not digits.isdigit():
        return None
    value = float(number)
    if value < 0:
        return None
    millis = value * unit
    if not millis.is_integer():
        return None
    return int(millis)
''',
}


MUTANTS = {
    "vare_ig.train.ini_sections.v1": REFERENCES["vare_ig.train.ini_sections.v1"].replace(
        'line.split("=", 1)', 'line.split("=")'
    ),
    "vare_ig.train.csv_record.v1": '''def solve(text):
    return text.split(",")
''',
    "vare_ig.train.cli_overrides.v1": REFERENCES["vare_ig.train.cli_overrides.v1"].replace(
        'result[key.replace("-", "_")] = False', 'result[key.replace("-", "_")] = "true"'
    ),
    "vare_ig.train.stable_toposort.v1": REFERENCES["vare_ig.train.stable_toposort.v1"].replace(
        'ready = sorted(node for node in nodes if indegree[node] == 0)',
        'ready = [node for node in nodes if indegree[node] == 0]',
    ).replace('ready.sort()', 'pass'),
    "vare_ig.train.graph_reachable.v1": REFERENCES["vare_ig.train.graph_reachable.v1"].replace(
        'return sorted(reached)', 'return sorted(node for node in reached if node in graph)'
    ),
    "vare_ig.train.undirected_components.v1": REFERENCES["vare_ig.train.undirected_components.v1"].replace(
        'adjacency[neighbor].add(node)', 'adjacency[neighbor].add(neighbor)'
    ),
    "vare_ig.train.merge_busy_windows.v1": REFERENCES["vare_ig.train.merge_busy_windows.v1"].replace(
        'start <= result[-1][1] + buffer', 'start < result[-1][1] + buffer'
    ),
    "vare_ig.train.running_key_totals.v1": REFERENCES["vare_ig.train.running_key_totals.v1"].replace(
        'totals[key] = totals.get(key, 0) + amount', 'totals[key] = amount'
    ),
    "vare_ig.train.latest_record_by_key.v1": REFERENCES["vare_ig.train.latest_record_by_key.v1"].replace(
        'timestamp >= latest[key][0]', 'timestamp > latest[key][0]'
    ),
    "vare_ig.train.deep_merge_defaults.v1": '''def solve(config, defaults):
    result = defaults.copy()
    result.update(config)
    return result
''',
    "vare_ig.train.normalize_relative_path.v1": REFERENCES["vare_ig.train.normalize_relative_path.v1"].replace(
        'return None\n            parts.pop()', 'parts.pop()'
    ),
    "vare_ig.train.parse_duration_ms.v1": REFERENCES["vare_ig.train.parse_duration_ms.v1"].replace(
        'if not millis.is_integer():', 'if False:'
    ),
}


def all_controls() -> tuple[dict[str, str], dict[str, str]]:
    references = dict(REFERENCES)
    mutants = dict(MUTANTS)
    references["vare_ig.explore.dedupe_event_ids.v1"] = '''def solve(events):
    seen = set()
    result = []
    for row in events:
        if row[0] not in seen:
            seen.add(row[0])
            result.append(row)
    return result
'''
    mutants["vare_ig.explore.dedupe_event_ids.v1"] = '''def solve(events):
    latest = {}
    for row in events:
        latest[row[0]] = row
    result = []
    seen = set()
    for row in events:
        key = row[0]
        if key not in seen:
            result.append(latest[key])
            seen.add(key)
    return result
'''
    same_contracts = {
        "vare_ig.explore.parse_csv_header.v1": "vare_ig.train.csv_record.v1",
        "vare_ig.explore.merge_event_ranges.v1": "vare_ig.train.merge_busy_windows.v1",
        "vare_ig.explore.running_event_counts.v1": "vare_ig.train.running_key_totals.v1",
        "vare_ig.explore.deep_defaults_config.v1": "vare_ig.train.deep_merge_defaults.v1",
        "vare_ig.explore.duration_ms.v1": "vare_ig.train.parse_duration_ms.v1",
        "vare_ig.explore.relative_path.v1": "vare_ig.train.normalize_relative_path.v1",
        "vare_ig.dev.parse_duration_pair.v1": "vare_ig.train.parse_duration_ms.v1",
        "vare_ig.dev.lexical_course_order.v1": "vare_ig.train.stable_toposort.v1",
        "vare_ig.dev.first_event_ids.v1": "vare_ig.explore.dedupe_event_ids.v1",
    }
    for new_id, old_id in same_contracts.items():
        references[new_id] = references[old_id]
        mutants[new_id] = mutants[old_id]
    references.update({
        "vare_ig.explore.ini_overrides.v1": '''def solve(lines):
    result = {}
    for line in lines:
        text = line.strip()
        if not text or text.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key:
            result[key] = value
    return result
''',
        "vare_ig.explore.graph_course_order.v1": '''def solve(courses, prerequisites):
    nodes = sorted(set(courses))
    node_set = set(nodes)
    outgoing = {node: set() for node in nodes}
    indegree = {node: 0 for node in nodes}
    for course, prerequisite in prerequisites:
        if course not in node_set or prerequisite not in node_set:
            return None
        if course not in outgoing[prerequisite]:
            outgoing[prerequisite].add(course)
            indegree[course] += 1
    ready = sorted(node for node in nodes if indegree[node] == 0)
    order = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for child in sorted(outgoing[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    return order if len(order) == len(nodes) else None
''',
        "vare_ig.explore.shortest_unweighted_path.v1": '''def solve(graph, start, goal):
    if start == goal:
        return [start]
    pending = [[start]]
    visited = {start}
    while pending:
        path = pending.pop(0)
        for neighbor in sorted(graph.get(path[-1], [])):
            if neighbor in visited:
                continue
            next_path = path + [neighbor]
            if neighbor == goal:
                return next_path
            visited.add(neighbor)
            pending.append(next_path)
    return None
''',
        "vare_ig.dev.deep_lookup.v1": '''def solve(config, path, default):
    value = config
    if not path:
        return config
    for part in path.split("."):
        if type(value) is not dict or part not in value:
            return default
        value = value[part]
    return value
''',
    })
    mutants.update({
        "vare_ig.explore.ini_overrides.v1": references["vare_ig.explore.ini_overrides.v1"].replace(
            'line.split("=", 1)', 'line.split("=")'
        ),
        "vare_ig.explore.graph_course_order.v1": '''def solve(courses, prerequisites):
    nodes = sorted(set(courses))
    outgoing = {node: set() for node in nodes}
    indegree = {node: 0 for node in nodes}
    for course, prerequisite in prerequisites:
        if course not in outgoing or prerequisite not in outgoing:
            return None
        if prerequisite not in outgoing[course]:
            outgoing[course].add(prerequisite)
            indegree[prerequisite] += 1
    ready = sorted(node for node in nodes if indegree[node] == 0)
    order = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for child in sorted(outgoing[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    return order if len(order) == len(nodes) else None
''',
        "vare_ig.explore.shortest_unweighted_path.v1": '''def solve(graph, start, goal):
    if start == goal:
        return [start]
    pending = [[start]]
    visited = {start}
    while pending:
        path = pending.pop()
        if path[-1] == goal:
            return path
        for neighbor in sorted(graph.get(path[-1], []), reverse=True):
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(path + [neighbor])
    return None
''',
        "vare_ig.explore.merge_event_ranges.v1": references["vare_ig.explore.merge_event_ranges.v1"].replace(
            'start <= result[-1][1] + buffer', 'start < result[-1][1] + buffer'
        ),
        "vare_ig.explore.running_event_counts.v1": references["vare_ig.explore.running_event_counts.v1"].replace(
            'totals[key] = totals.get(key, 0) + amount', 'totals[key] = amount'
        ),
        "vare_ig.explore.deep_defaults_config.v1": '''def solve(config, defaults):
    result = defaults.copy()
    result.update(config)
    return result
''',
        "vare_ig.explore.duration_ms.v1": references["vare_ig.explore.duration_ms.v1"].replace(
            'if not millis.is_integer():', 'if False:'
        ),
        "vare_ig.explore.relative_path.v1": references["vare_ig.explore.relative_path.v1"].replace(
            'return None\n            parts.pop()', 'parts.pop()'
        ),
        "vare_ig.dev.deep_lookup.v1": '''def solve(config, path, default):
    value = config
    if not path:
        return config
    for part in path.split("."):
        if type(value) is not dict or not value.get(part):
            return default
        value = value[part]
    return value
''',
    })
    # Reuse corresponding known-bad controls for shared Dev contracts.
    for new_id, old_id in {
        "vare_ig.dev.parse_duration_pair.v1": "vare_ig.train.parse_duration_ms.v1",
        "vare_ig.dev.lexical_course_order.v1": "vare_ig.train.stable_toposort.v1",
        "vare_ig.dev.first_event_ids.v1": "vare_ig.explore.dedupe_event_ids.v1",
    }.items():
        mutants[new_id] = mutants[old_id]
    return references, mutants


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite oracle audit: {output}")
    references, mutants = all_controls()
    expected_ids = {task.task_id for task in tasks.all_tasks()}
    if set(references) != expected_ids or set(mutants) != expected_ids:
        raise RuntimeError("every locked VARE-IG task needs one reference and one incorrect-mutant control")
    started = time.monotonic()
    results = []
    for task in tasks.all_tasks():
        reference = runner.score_source(task, references[task.task_id])
        mutant = runner.score_source(task, mutants[task.task_id])
        results.append({
            "task_id": task.task_id,
            "task_sha256": task.sha256(),
            "reference_source_sha256": hashlib.sha256(references[task.task_id].encode()).hexdigest(),
            "mutant_source_sha256": hashlib.sha256(mutants[task.task_id].encode()).hexdigest(),
            "reference_passed": reference["passed"],
            "reference_case_count": len(reference["cases"]),
            "reference_failed_cases": [case["case_id"] for case in reference["cases"] if not case["passed"]],
            "mutant_passed": mutant["passed"],
            "mutant_failed_cases": [case["case_id"] for case in mutant["cases"] if not case["passed"]],
        })
    report = {
        "schema": "vare_ig_task_oracle_mutant_audit_v1",
        "task_pack_sha256": hashlib.sha256(
            (ROOT / "src/vare/learning_pipeline/vare_ig_tasks.py").read_bytes()
        ).hexdigest(),
        "runner_sha256": hashlib.sha256((ROOT / "scripts/qwen3_function_repair_grpo.py").read_bytes()).hexdigest(),
        "candidate_worker_sha256": hashlib.sha256((ROOT / "src/vare/isolated_candidate.py").read_bytes()).hexdigest(),
        "container_image": runner.CONTAINER_IMAGE,
        "network_disabled": True,
        "task_count": len(results),
        "reference_pass_count": sum(bool(row["reference_passed"]) for row in results),
        "mutant_rejection_count": sum(not bool(row["mutant_passed"]) for row in results),
        "all_controls_passed": all(row["reference_passed"] and not row["mutant_passed"] for row in results),
        "wall_seconds": round(time.monotonic() - started, 6),
        "items": results,
        "claim_boundary": "isolated reference/mutant controls for these fixed VARE-IG function contracts only; not arbitrary repository grader security",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(output), "references_passed": report["reference_pass_count"],
                      "mutants_rejected": report["mutant_rejection_count"],
                      "all_controls_passed": report["all_controls_passed"]}, sort_keys=True))
    return 0 if report["all_controls_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
