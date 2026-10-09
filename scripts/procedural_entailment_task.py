#!/usr/bin/env python3
"""Deterministic procedural binary-entailment task used by a frozen CPU study."""
from __future__ import annotations

import hashlib
from typing import Any


TRAIN_STYLE = 0
DEV_STYLE = 1
CONFIRM_STYLE = 2
SHIFT_STYLE = 3

NAMES = (
    ("Ari", "Mina", "Sol", "Rin", "Tavi", "Noa", "Kiri", "Omi"),
    ("Ira", "Lena", "Miro", "Sana", "Toma", "Neri", "Vela", "Oru"),
    ("Ena", "Kato", "Lumi", "Ravi", "Selo", "Yuna", "Daro", "Mika"),
    ("Pali", "Rumi", "Teno", "Vani", "Kelo", "Zima", "Orin", "Faro"),
)

PREDICATES = (
    ("dax", "wug", "blicket", "toma", "zorp", "fep", "koba", "lunt", "niff", "veem", "pim", "sarl"),
    ("mep", "siv", "plone", "tav", "gorp", "nusk", "vab", "rilt", "yem", "poga", "kesh", "loba"),
    ("krel", "navi", "tupp", "bex", "mora", "zint", "lavo", "drem", "puk", "sarn", "fimo", "grel"),
    ("vot", "jasp", "nelo", "cav", "wemp", "rusk", "falo", "zeb", "dov", "mirt", "timo", "narp"),
)

TEMPLATES = {
    TRAIN_STYLE: (
        "All rules below are true. If a person is {left}, they are {right}.",
        "{name} is {start}. Using only these rules, does {name} necessarily have property {goal}? Answer with one word: Yes or No.",
    ),
    DEV_STYLE: (
        "Assume the following implications always hold: Anyone who is {left} must also be {right}.",
        "Known fact: {name} has property {start}. Check whether {name} must have property {goal}. Reply only Yes or No.",
    ),
    CONFIRM_STYLE: (
        "True statements: Being {left} implies being {right}.",
        "Fact: {name} is {start}. Claim: {name} is {goal}. Does the claim follow from the statements? Give exactly Yes or No.",
    ),
    SHIFT_STYLE: (
        "For every individual, property {left} guarantees property {right}.",
        "We know {name} has property {start}. Is it logically necessary that {name} has property {goal}? Answer Yes or No only.",
    ),
}


def _digest(seed: int, *parts: Any) -> bytes:
    payload = "\0".join(("vare-procedural-entailment-v1", str(seed), *(str(x) for x in parts)))
    return hashlib.sha256(payload.encode("utf-8")).digest()


def _u64(seed: int, *parts: Any) -> int:
    return int.from_bytes(_digest(seed, *parts)[:8], "big")


def _unit(seed: int, *parts: Any) -> float:
    return _u64(seed, *parts) / 18446744073709551616.0


def _pick(values, seed: int, *parts: Any):
    return values[_u64(seed, *parts) % len(values)]


def _graph(seed: int, index: int, n_nodes: int, edge_probability: float):
    for attempt in range(1000):
        edges = []
        for left in range(n_nodes):
            for right in range(left + 1, n_nodes):
                if _unit(seed, index, attempt, "edge", left, right) < edge_probability:
                    edges.append((left, right))
        start = _u64(seed, index, attempt, "start") % n_nodes
        reached = {start}
        changed = True
        while changed:
            changed = False
            for left, right in edges:
                if left in reached and right not in reached:
                    reached.add(right)
                    changed = True
        reachable = sorted(reached - {start})
        unreachable = sorted(set(range(n_nodes)) - reached)
        if reachable and unreachable:
            return edges, start, reachable, unreachable
    raise RuntimeError("could not construct a graph with reachable and unreachable goals")


def _balanced_labels(seed: int, count: int, positives: int):
    ranked = sorted(range(count), key=lambda i: (_digest(seed, "label-allocation", i), i))
    yes = set(ranked[:positives])
    return ["Yes" if i in yes else "No" for i in range(count)]


def generate_split(seed: int, count: int, style: int, positive_count: int, n_nodes: int, edge_probability: float):
    if count < 2 or not 0 < positive_count < count or style not in TEMPLATES:
        raise ValueError("invalid split configuration")
    names = NAMES[style]
    predicates = PREDICATES[style]
    prefix, suffix = TEMPLATES[style]
    labels = _balanced_labels(seed, count, positive_count)
    rows = []
    for index, label in enumerate(labels):
        edges, start, reachable, unreachable = _graph(seed, index, n_nodes, edge_probability)
        goal_choices = reachable if label == "Yes" else unreachable
        goal = _pick(goal_choices, seed, index, "goal")
        entity = _pick(names, seed, index, "name")
        # Remap graph indices with a deterministic permutation so the conclusion
        # label is not coupled to a fixed lexical or numeric direction.
        order = sorted(range(n_nodes), key=lambda node: (_digest(seed, index, "predicate-order", node), node))
        if n_nodes > len(predicates):
            raise ValueError("the selected predicate vocabulary is too small for this graph")
        pred_for_node = {node: predicates[order[node]] for node in range(n_nodes)}
        rendered_edges = [(pred_for_node[a], pred_for_node[b]) for a, b in edges]
        rendered_edges.sort(key=lambda pair: (_digest(seed, index, "rule-order", pair[0], pair[1]), pair))
        rules = [prefix.format(left=left, right=right) for left, right in rendered_edges]
        rules.append(suffix.format(name=entity, start=pred_for_node[start], goal=pred_for_node[goal]))
        prompt = "\n".join(rules)
        mapping = [pred_for_node[node] for node in range(n_nodes)]
        rows.append({
            "index": index,
            "style": style,
            "label": label,
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "graph": {
                "n_nodes": n_nodes,
                "edges": [list(edge) for edge in edges],
                "start": start,
                "goal": goal,
                "reachable_nodes": sorted({start, *reachable}),
                "edge_probability": edge_probability,
                "predicate_names_by_node": mapping,
                "entity": entity,
                "rendered_edges": [list(edge) for edge in rendered_edges],
            },
        })
    order = sorted(range(count), key=lambda i: (_digest(seed, "presentation-order", i), i))
    rows = [rows[i] for i in order]
    for index, row in enumerate(rows):
        row["presentation_index"] = index
    verify_split(rows, count, positive_count)
    return rows


def verify_split(rows, expected_count: int, expected_positive_count: int):
    if len(rows) != expected_count:
        raise ValueError("wrong split size")
    hashes = [row["prompt_sha256"] for row in rows]
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate prompts within split")
    if sum(row["label"] == "Yes" for row in rows) != expected_positive_count:
        raise ValueError("wrong positive-label count")
    for row in rows:
        if hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest() != row["prompt_sha256"]:
            raise ValueError("prompt hash does not match rendered prompt")
        graph = row["graph"]
        edges = {tuple(edge) for edge in graph["edges"]}
        if any(len(edge) != 2 or not all(0 <= node < graph["n_nodes"] for node in edge)
               for edge in graph["edges"]):
            raise ValueError("graph contains an invalid edge")
        if not (0 <= graph["start"] < graph["n_nodes"] and
                0 <= graph["goal"] < graph["n_nodes"] and graph["goal"] != graph["start"]):
            raise ValueError("graph contains an invalid start or goal")
        names = graph["predicate_names_by_node"]
        if len(names) != graph["n_nodes"] or len(set(names)) != len(names):
            raise ValueError("predicate mapping is incomplete or non-unique")
        expected_rendered_edges = {(names[left], names[right]) for left, right in edges}
        actual_rendered_edges = {tuple(edge) for edge in graph["rendered_edges"]}
        if actual_rendered_edges != expected_rendered_edges:
            raise ValueError("rendered rules do not encode the graph edges")
        prefix, suffix = TEMPLATES[row["style"]]
        expected_prompt = "\n".join(
            [prefix.format(left=left, right=right) for left, right in graph["rendered_edges"]]
            + [suffix.format(name=graph["entity"], start=names[graph["start"]], goal=names[graph["goal"]])]
        )
        if row["prompt"] != expected_prompt:
            raise ValueError("prompt text does not encode the stored graph and query")
        reached = {graph["start"]}
        while True:
            expanded = reached | {right for left, right in edges if left in reached}
            if expanded == reached:
                break
            reached = expanded
        expected = "Yes" if graph["goal"] in reached else "No"
        if row["label"] != expected:
            raise ValueError("label disagrees with graph-reachability oracle")
        if sorted(reached) != graph["reachable_nodes"]:
            raise ValueError("recorded reachable-node proof does not match graph closure")
