"""Exact finite feasibility oracle for keeping rollout groups within one microbatch."""

from functools import lru_cache
import json


FIXTURES = {
    "fixed_count": {
        "FC1": {"processes": 2, "capacity": 4, "groups": [[2], [1], [1]]},
        "FC2": {"processes": 2, "capacity": 4, "groups": [[3], [2]]},
        "FC3": {"processes": 2, "capacity": 4, "groups": [[5]]},
        "FC4": {"processes": 2, "capacity": 4, "groups": [[1], [1]]},
    },
    "token_budget": {
        "TB1": {"processes": 2, "capacity": 10, "groups": [[6, 4], [6]]},
        "TB2": {"processes": 2, "capacity": 10, "groups": [[6, 6], [4, 4]]},
        "TB3": {"processes": 2, "capacity": 10, "groups": [[4], [5]]},
        "TB4": {"processes": 2, "capacity": 10, "groups": [[6, 6, 6]]},
        "TB5": {"processes": 2, "capacity": 10, "groups": [[6, 6], [5, 5]]},
        "TB6": {"processes": 2, "capacity": 10, "groups": [[6, 5], [4, 4]]},
    },
}

EXPECTED = {
    "FC1": (1, []), "FC2": (2, []), "FC3": (0, [0]), "FC4": (1, []),
    "TB1": (1, []), "TB2": (1, []), "TB3": (1, []),
    "TB4": (0, [0]), "TB5": (2, []), "TB6": (1, []),
}


def assign_bins(lengths, processes, budget, require_nonempty=True):
    """Return one exact assignment or None; sort/branch ordering affects speed, not feasibility."""
    if any(length < 1 or length > budget for length in lengths):
        return None
    ordered = sorted(enumerate(lengths), key=lambda pair: pair[1], reverse=True)
    loads = [0] * processes
    rows = [[] for _ in range(processes)]

    def place(index):
        if index == len(ordered):
            return not require_nonempty or all(rows)
        original_index, length = ordered[index]
        seen_loads = set()
        for process in range(processes):
            if loads[process] in seen_loads:
                continue
            seen_loads.add(loads[process])
            if loads[process] + length > budget:
                continue
            loads[process] += length
            rows[process].append(original_index)
            result = place(index + 1)
            if result:
                return result
            rows[process].pop()
            loads[process] -= length
        return None

    result = place(0)
    if result is None:
        return None
    return {"row_indices": rows, "row_tokens": loads}


def fixed_segment(groups, processes, capacity):
    count = sum(map(len, groups))
    if count < processes or count > capacity:
        return None
    lengths = [length for group in groups for length in group]
    # Fixed-count mode has no per-row token cap; use longest-first squared-load balancing.
    rows = [[] for _ in range(processes)]
    loads = [0] * processes
    for index in sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True):
        target = min(range(processes), key=lambda p: (loads[p], p))
        rows[target].append(index)
        loads[target] += lengths[index] ** 2
    return {"row_indices": rows, "row_tokens": [sum(lengths[i] for i in row) for row in rows]}


def token_segment(groups, processes, budget):
    lengths = [length for group in groups for length in group]
    return assign_bins(lengths, processes, budget, require_nonempty=True)


def partition(groups, segmenter):
    """Find a contiguous whole-group partition into feasible microbatches."""

    @lru_cache(None)
    def solve(start):
        if start == len(groups):
            return ()
        for end in range(len(groups), start, -1):
            assignment = segmenter(groups[start:end])
            if assignment is None:
                continue
            tail = solve(end)
            if tail is not None:
                return ((start, end, assignment),) + tail
        return None

    return solve(0)


def run_case(mode, name, case):
    processes = case["processes"]
    capacity = case["capacity"]
    groups = case["groups"]
    if mode == "fixed_count":
        rejected = [i for i, group in enumerate(groups) if len(group) > capacity]
        retained = [(i, group) for i, group in enumerate(groups) if i not in rejected]
        groups = [group for _, group in retained]
        segmenter = lambda segment: fixed_segment(segment, processes, capacity)
    else:
        rejected = []
        retained = []
        for i, group in enumerate(groups):
            intrinsically_impossible = any(length > capacity for length in group) or sum(group) > processes * capacity
            if not intrinsically_impossible and len(group) >= processes:
                intrinsically_impossible = assign_bins(group, processes, capacity, require_nonempty=False) is None
            if intrinsically_impossible:
                rejected.append(i)
            else:
                retained.append((i, group))
        groups = [group for _, group in retained]
        segmenter = lambda segment: token_segment(segment, processes, capacity)

    result = partition(groups, segmenter) if groups else ()
    if result is None:
        raise AssertionError(f"{name}: no valid contiguous partition")

    assigned_group_batches = []
    for start, end, assignment in result:
        row_count = sum(len(group) for group in groups[start:end])
        flattened_indices = [index for row in assignment["row_indices"] for index in row]
        if sorted(flattened_indices) != list(range(row_count)):
            raise AssertionError(f"{name}: row assignment lost or duplicated an input")
        if len(assignment["row_indices"]) != processes or any(not row for row in assignment["row_indices"]):
            raise AssertionError(f"{name}: emitted an empty process row")
        if mode == "token_budget" and any(tokens > capacity for tokens in assignment["row_tokens"]):
            raise AssertionError(f"{name}: exceeded per-row token budget")
        assigned_group_batches.append([retained[i][0] for i in range(start, end)])

    retained_group_ids = [original_index for original_index, _ in retained]
    if [group_id for batch in assigned_group_batches for group_id in batch] != retained_group_ids:
        raise AssertionError(f"{name}: a retained rollout was lost, duplicated, or reordered")
    expected_batch_count, expected_rejected = EXPECTED[name]
    if len(assigned_group_batches) != expected_batch_count or rejected != expected_rejected:
        raise AssertionError(f"{name}: result differed from the locked case disposition")

    retained_count = sum(len(group) for group in groups)
    rejected_count = sum(len(case["groups"][i]) for i in rejected)
    result_json = {
        "case": name,
        "microbatches": assigned_group_batches,
        "rejected_group_indices": rejected,
        "retained_row_count": retained_count,
        "rejected_row_count": rejected_count,
        "capacity_respected": True,
        "whole_group_accounting": True,
    }
    result_json["row_tokens"] = [item[2]["row_tokens"] for item in result]
    return result_json


def main():
    outputs = {}
    for mode, cases in FIXTURES.items():
        outputs[mode] = [run_case(mode, name, case) for name, case in cases.items()]
    print(json.dumps(outputs, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
