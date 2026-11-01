"""Prospective, parent-scored task contracts for the VARE-IG pilot.

The task set is synthetic and intended for learning-dynamics and systems
measurement. Expected values live in this parent-side module and are never
included in model prompts. It is versioned independently from the retired
six-task function-repair study.
"""
from __future__ import annotations

from vare.function_repair_tasks import ContractCase, FunctionRepairTask


TRAIN_TASKS = (
    FunctionRepairTask(
        "vare_ig.train.ini_sections.v1", "train",
        "Implement `solve(lines)`. Parse a small INI-like format into a dictionary mapping section names to key/value dictionaries. "
        "The initial section name is the empty string. A trimmed line beginning with `#` or `;`, a blank line, or a malformed "
        "section header is ignored. A section header is a trimmed line exactly like `[name]` with a nonempty trimmed name. "
        "Assignments split at the first `=`; trim key and value, ignore empty keys, and let later assignments replace earlier ones. "
        "Reopening a section keeps its previous keys. Values are otherwise unchanged.",
        (
            ContractCase("initial_and_reopened", (["A=1", "[db]", "host = localhost", "port=5432", "[db]", "host=prod"],),
                         {"": {"A": "1"}, "db": {"host": "prod", "port": "5432"}}),
            ContractCase("comments_and_blanks", (["# top", "; also comment", "", "[ app ]", " # inner", "x = y"],),
                         {"app": {"x": "y"}}),
            ContractCase("first_equals_only", (["[x]", "url=a=b=c", "=skip"],), {"x": {"url": "a=b=c"}}),
            ContractCase("malformed_section", (["[broken", "k=v", "[]", "z=2"],), {"": {"k": "v", "z": "2"}}),
            ContractCase("empty", ([],), {}),
        ),
    ),
    FunctionRepairTask(
        "vare_ig.train.csv_record.v1", "train",
        "Implement `solve(text)` for one comma-separated record. A quoted field begins with `\"` at the first character of that field, "
        "ends at the next unescaped `\"`, and represents an embedded quote as `\"\"`. Commas inside quoted fields are data. "
        "Unquoted fields are copied exactly, including whitespace. After a closing quote only a comma or end-of-input is allowed. "
        "A quote inside an unquoted field, an unclosed quote, or other text after a closing quote makes the record malformed; return `None`. "
        "The empty record is one empty field. Return a list of strings.",
        (
            ContractCase("plain", ("a,b,c",), ["a", "b", "c"]),
            ContractCase("quoted_comma", ('"a,b",c',), ["a,b", "c"]),
            ContractCase("doubled_quote", ('a,"b""c",""',), ["a", 'b"c', ""]),
            ContractCase("empty_record", ("",), [""]),
            ContractCase("malformed_after_quote", ('a,"b"x',), None),
        ),
    ),
    FunctionRepairTask(
        "vare_ig.train.cli_overrides.v1", "train",
        "Implement `solve(tokens)`. Parse command-line override tokens into a dictionary. Ignore positional tokens and tokens not beginning "
        "with `--`. Convert hyphens to underscores in option names. `--key=value` stores the text after the first `=`. `--key value` "
        "consumes the next token as its value when it does not begin with `--`; otherwise `--key` stores the string `\"true\"` and does not "
        "consume the next option. `--no-key` stores boolean `False`. An empty key is ignored. Later occurrences replace earlier ones.",
        (
            ContractCase("mixed_forms", (["run", "--host", "localhost", "--port=8080", "--debug", "--no-cache"],),
                         {"host": "localhost", "port": "8080", "debug": "true", "cache": False}),
            ContractCase("repeated_and_split_once", (["--name=old", "--name", "new=value"],), {"name": "new=value"}),
            ContractCase("does_not_consume_option", (["--alpha", "--beta=x"],), {"alpha": "true", "beta": "x"}),
            ContractCase("empty_key_and_positionals", (["word", "--=x", "---odd=value"],), {"_odd": "value"}),
            ContractCase("empty", ([],), {}),
        ),
    ),
    FunctionRepairTask(
        "vare_ig.train.stable_toposort.v1", "train",
        "Implement `solve(nodes, edges)`. Return the lexicographically smallest topological ordering of unique string nodes. "
        "Each edge is `[before, after]`; duplicate edges are ignored. If an edge refers to a node absent from `nodes`, or if the "
        "graph contains a directed cycle, return `None`. Include isolated nodes. The input lists contain strings.",
        (
            ContractCase("lexical_ready_set", (["b", "a", "c", "d"], [["b", "a"], ["b", "d"]]),
                         ["b", "a", "c", "d"]),
            ContractCase("duplicate_nodes_and_edges", (["a", "b", "a"], [["a", "b"], ["a", "b"]]), ["a", "b"]),
            ContractCase("isolated", (["z", "a"], []), ["a", "z"]),
            ContractCase("cycle", (["a", "b"], [["a", "b"], ["b", "a"]]), None),
            ContractCase("unknown_endpoint", (["a"], [["a", "missing"]]), None),
        ),
    ),
    FunctionRepairTask(
        "vare_ig.train.graph_reachable.v1", "train",
        "Implement `solve(graph, start)`. `graph` maps string node names to lists of neighboring string names. Treat edges as directed. "
        "Return all nodes reachable from `start`, including `start` if it exists in the graph, sorted lexicographically. A neighbor not "
        "listed as a graph key is still a reachable node with no outgoing edges. If `start` is absent both as a key and as a neighbor, "
        "return `[]`. Cycles must terminate.",
        (
            ContractCase("diamond", ({"a": ["c", "b"], "b": ["d"], "c": ["d"], "d": []}, "a"), ["a", "b", "c", "d"]),
            ContractCase("cycle", ({"a": ["b"], "b": ["a", "c"]}, "a"), ["a", "b", "c"]),
            ContractCase("unknown_start", ({"a": ["b"]}, "z"), []),
            ContractCase("leaf_key", ({"a": []}, "a"), ["a"]),
            ContractCase("neighbor_without_key", ({"a": ["z"]}, "a"), ["a", "z"]),
        ),
    ),
    FunctionRepairTask(
        "vare_ig.train.undirected_components.v1", "train",
        "Implement `solve(graph)`. The input maps string node names to neighbor lists and describes an undirected graph: every listed "
        "relationship connects both endpoints even if only one direction is present. Include nodes appearing only as neighbors. Return "
        "connected components as sorted lists, with components sorted by their first (smallest) member. Duplicate edges and self-edges "
        "must not duplicate nodes. Return `[]` for an empty graph.",
        (
            ContractCase("two_components", ({"a": ["b"], "b": [], "c": ["d"], "d": []},), [["a", "b"], ["c", "d"]]),
            ContractCase("one_sided_and_isolated", ({"m": ["a"], "z": []},), [["a", "m"], ["z"]]),
            ContractCase("duplicates_and_self", ({"a": ["a", "b", "b"], "b": []},), [["a", "b"]]),
            ContractCase("empty", ({},), []),
            ContractCase("neighbor_only", ({"root": ["leaf"]},), [["leaf", "root"]]),
        ),
    ),
    FunctionRepairTask(
        "vare_ig.train.merge_busy_windows.v1", "train",
        "Implement `solve(windows, buffer)`. Each window is a closed integer interval `[start, end]` with `start <= end`. Sort by start "
        "and merge overlapping windows or windows whose gap is at most `buffer`; the merged end is the maximum end. Keep windows "
        "separate when the gap is larger. If `buffer < 0`, return `None`. Do not mutate the input. Return a list of intervals.",
        (
            ContractCase("buffer_merges_gap", ([[1, 3], [4, 5], [9, 10]], 1), [[1, 5], [9, 10]]),
            ContractCase("overlap_and_unsorted", ([[5, 8], [1, 4], [3, 6]], 0), [[1, 8]]),
            ContractCase("no_merge", ([[1, 2], [4, 5]], 1), [[1, 2], [4, 5]]),
            ContractCase("empty", ([], 2), []),
            ContractCase("negative_buffer", ([[1, 2]], -1), None),
        ),
    ),
    FunctionRepairTask(
        "vare_ig.train.running_key_totals.v1", "train",
        "Implement `solve(events)`. Each event is `[key, integer_amount]`. Return one `[key, cumulative_total]` row per event in the "
        "original order, where each key has an independent running total. Negative amounts are allowed. Do not mutate the input. "
        "Return `[]` for empty input.",
        (
            ContractCase("interleaved", ([["a", 2], ["b", 1], ["a", -1]],), [["a", 2], ["b", 1], ["a", 1]]),
            ContractCase("negative_first", ([["x", -3], ["x", 5]],), [["x", -3], ["x", 2]]),
            ContractCase("single", ([["only", 7]],), [["only", 7]]),
            ContractCase("empty", ([],), []),
            ContractCase("zero_and_revisit", ([["a", 0], ["b", 4], ["a", 2], ["b", -4]],),
                         [["a", 0], ["b", 4], ["a", 2], ["b", 0]]),
        ),
    ),
    FunctionRepairTask(
        "vare_ig.train.latest_record_by_key.v1", "train",
        "Implement `solve(records)`. Each record is `[key, value, timestamp]`, where key is a string and timestamp is an integer. "
        "For each key, keep the record with the greatest timestamp; if timestamps tie, keep the record that appeared later in the "
        "input. Return the original winning records sorted by key. Return `[]` for empty input.",
        (
            ContractCase("newer_and_later_tie", ([["b", "old", 1], ["a", "x", 2], ["b", "new", 3], ["a", "tie", 2]],),
                         [["a", "tie", 2], ["b", "new", 3]]),
            ContractCase("out_of_order", ([["x", "late", 9], ["x", "early", 2]],), [["x", "late", 9]]),
            ContractCase("distinct", ([["z", 1, 0], ["a", 2, 0]],), [["a", 2, 0], ["z", 1, 0]]),
            ContractCase("empty", ([],), []),
            ContractCase("same_time_last_wins", ([["k", "first", 4], ["k", "second", 4]],), [["k", "second", 4]]),
        ),
    ),
    FunctionRepairTask(
        "vare_ig.train.deep_merge_defaults.v1", "train",
        "Implement `solve(config, defaults)`. Recursively add missing keys from `defaults` to `config`. When both values are dictionaries, "
        "merge recursively. Any value already present in `config`, including `None`, a list, or a scalar where the default is a dictionary, "
        "wins and is preserved. Do not mutate either input. Return the merged dictionary.",
        (
            ContractCase("nested_fill", ({"db": {"host": "prod", "flags": {"tls": True}}},
                                          {"db": {"host": "localhost", "port": 5432, "flags": {"tls": False, "verify": True}},
                                           "debug": False}),
                         {"db": {"host": "prod", "port": 5432, "flags": {"tls": True, "verify": True}}, "debug": False}),
            ContractCase("explicit_null_wins", ({"db": None}, {"db": {"port": 5432}}), {"db": None}),
            ContractCase("lists_are_replaced_not_merged", ({"items": [1]}, {"items": [2, 3]}), {"items": [1]}),
            ContractCase("missing_branch", ({}, {"a": {"b": 1}}), {"a": {"b": 1}}),
            ContractCase("empty", ({}, {}), {}),
        ),
    ),
    FunctionRepairTask(
        "vare_ig.train.normalize_relative_path.v1", "train",
        "Implement `solve(path)`. Normalize a POSIX-style relative path by removing empty and `.` segments and resolving `..` against a "
        "previous ordinary segment. If resolving `..` would move above the starting directory, or if the path starts with `/`, return `None`. "
        "Return the remaining segments joined by `/`; an empty normalized path is `\"\"`. Backslashes are ordinary characters, not separators.",
        (
            ContractCase("internal_parent", ("lib/./pkg/../x.py",), "lib/x.py"),
            ContractCase("empty_result", ("./a/..",), ""),
            ContractCase("collapse_slashes", ("a//b///c",), "a/b/c"),
            ContractCase("absolute_rejected", ("/etc/passwd",), None),
            ContractCase("escape_rejected", ("a/../../b",), None),
        ),
    ),
    FunctionRepairTask(
        "vare_ig.train.parse_duration_ms.v1", "train",
        "Implement `solve(text)`. Parse a nonnegative decimal duration with suffix `ms`, `s`, `m`, or `h`, converting it to integer "
        "milliseconds. Allow surrounding whitespace and decimal quantities, but return `None` if the value is negative, malformed, has an "
        "unknown suffix, or does not represent a whole number of milliseconds. Do not use rounding. Examples of units: 1 s = 1000 ms, "
        "1 m = 60000 ms, and 1 h = 3600000 ms.",
        (
            ContractCase("seconds_fraction", (" 1.5s ",), 1500),
            ContractCase("milliseconds", ("250ms",), 250),
            ContractCase("minutes", ("2m",), 120000),
            ContractCase("fractional_hour", ("0.25h",), 900000),
            ContractCase("fractional_millisecond_rejected", ("0.1ms",), None),
        ),
    ),
)


EXPLORATION_TASKS = (
    FunctionRepairTask(
        "vare_ig.explore.parse_csv_header.v1", "exploration",
        "Implement `solve(text)`. Parse one CSV-like header with comma-separated fields, double-quoted fields, and doubled quotes inside "
        "quoted fields. Preserve unquoted whitespace. A quote inside an unquoted field, unclosed quote, or characters after a closing quote "
        "other than a comma make the input malformed; return `None`. Empty input returns `[\"\"]`.",
        (
            ContractCase("basic", ("name,age,city",), ["name", "age", "city"]),
            ContractCase("quoted", ('"last, first",age',), ["last, first", "age"]),
            ContractCase("escaped_quote", ('"say ""hi""",x',), ['say "hi"', "x"]),
            ContractCase("empty", ("",), [""]),
            ContractCase("bad_quote", ('a"b,c',), None),
        ),
    ),
    FunctionRepairTask(
        "vare_ig.explore.ini_overrides.v1", "exploration",
        "Implement `solve(lines)`. Read `key=value` settings from lines. Ignore blank lines and lines whose trimmed form begins with `#`. "
        "Split at the first equals sign, trim both parts, ignore empty keys, and use the last assignment for duplicate keys. Return a string "
        "dictionary.",
        (
            ContractCase("last_wins", (["mode=dev", "mode=prod"],), {"mode": "prod"}),
            ContractCase("first_equals", (["url=a=b"],), {"url": "a=b"}),
            ContractCase("trim", ([" x = y "],), {"x": "y"}),
            ContractCase("skip", (["", "#x", "no", "=bad"],), {}),
            ContractCase("empty", ([],), {}),
        ),
    ),
    FunctionRepairTask(
        "vare_ig.explore.graph_course_order.v1", "exploration",
        "Implement `solve(courses, prerequisites)`. A pair `[course, prerequisite]` means the prerequisite must come first. Return the "
        "lexicographically smallest valid course order, including all unique courses, or `None` if a cycle exists or a pair references an "
        "unknown course. Duplicate pairs are ignored.",
        (
            ContractCase("order", (["b", "a", "c"], [["c", "a"], ["c", "b"]]), ["a", "b", "c"]),
            ContractCase("independent", (["z", "a"], []), ["a", "z"]),
            ContractCase("cycle", (["a", "b"], [["a", "b"], ["b", "a"]]), None),
            ContractCase("duplicate", (["a", "b"], [["b", "a"], ["b", "a"]]), ["a", "b"]),
            ContractCase("unknown", (["a"], [["a", "x"]]), None),
        ),
    ),
    FunctionRepairTask(
        "vare_ig.explore.shortest_unweighted_path.v1", "exploration",
        "Implement `solve(graph, start, goal)`. `graph` maps string nodes to neighbor lists and describes a directed graph. Return a shortest "
        "path as a list including both endpoints; among shortest paths choose the lexicographically smallest full node sequence. Return `[start]` "
        "when start equals goal, and `None` when no path exists. A neighbor not present as a key is a leaf.",
        (
            ContractCase("shortest_beats_lexical_longer", ({"s": ["a", "b"], "a": ["c"], "c": ["t"], "b": ["t"], "t": []}, "s", "t"), ["s", "b", "t"]),
            ContractCase("lexical_tie", ({"s": ["b", "a"], "a": ["t"], "b": ["t"]}, "s", "t"), ["s", "a", "t"]),
            ContractCase("same_node", ({}, "x", "x"), ["x"]),
            ContractCase("missing", ({"a": ["b"]}, "a", "z"), None),
            ContractCase("cycle", ({"a": ["b"], "b": ["a", "t"]}, "a", "t"), ["a", "b", "t"]),
        ),
    ),
    FunctionRepairTask(
        "vare_ig.explore.merge_event_ranges.v1", "exploration",
        "Implement `solve(events, gap)`. Each event is `[start, end]` with integer endpoints. Sort intervals by start and merge overlap or "
        "gaps no larger than `gap`; use the maximum end. A negative gap returns `None`. Return `[]` for no events.",
        (
            ContractCase("gap", ([[1, 2], [4, 5]], 2), [[1, 5]]),
            ContractCase("separate", ([[1, 2], [5, 7]], 2), [[1, 2], [5, 7]]),
            ContractCase("nested", ([[1, 10], [3, 4], [8, 12]], 0), [[1, 12]]),
            ContractCase("empty", ([], 0), []),
            ContractCase("negative", ([[1, 2]], -1), None),
        ),
    ),
    FunctionRepairTask(
        "vare_ig.explore.running_event_counts.v1", "exploration",
        "Implement `solve(events)`. Each event is `[key, amount]`. Return cumulative totals per key after each event, preserving input order. "
        "Amounts may be negative. Return `[]` for empty input.",
        (
            ContractCase("interleaved", ([["a", 1], ["b", 3], ["a", 2]],), [["a", 1], ["b", 3], ["a", 3]]),
            ContractCase("negative", ([["x", -2], ["x", -2]],), [["x", -2], ["x", -4]]),
            ContractCase("empty", ([],), []),
            ContractCase("zero", ([["z", 0]],), [["z", 0]]),
            ContractCase("revisit", ([["a", 4], ["b", 2], ["a", 1]],), [["a", 4], ["b", 2], ["a", 5]]),
        ),
    ),
    FunctionRepairTask(
        "vare_ig.explore.dedupe_event_ids.v1", "exploration",
        "Implement `solve(events)`. Each event is `[event_id, payload]`. Keep the first occurrence of each string event ID and preserve "
        "the order of those first occurrences. Return the retained original rows without mutating input.",
        (
            ContractCase("first_wins", ([["a", 1], ["b", 2], ["a", 3]],), [["a", 1], ["b", 2]]),
            ContractCase("order", ([["z", 0], ["a", 1], ["m", 2]],), [["z", 0], ["a", 1], ["m", 2]]),
            ContractCase("empty", ([],), []),
            ContractCase("all_duplicate", ([["x", 1], ["x", 2], ["x", 3]],), [["x", 1]]),
            ContractCase("case_sensitive", ([["A", 1], ["a", 2]],), [["A", 1], ["a", 2]]),
        ),
    ),
    FunctionRepairTask(
        "vare_ig.explore.deep_defaults_config.v1", "exploration",
        "Implement `solve(config, defaults)`. Recursively fill missing dictionary keys from defaults; values already present in config always "
        "win, including `None` and lists. Do not mutate inputs.",
        (
            ContractCase("nested", ({"db": {"host": "prod"}}, {"db": {"host": "local", "port": 2}, "debug": False}),
                         {"db": {"host": "prod", "port": 2}, "debug": False}),
            ContractCase("null", ({"x": None}, {"x": 1}), {"x": None}),
            ContractCase("list", ({"x": [1]}, {"x": [2]}), {"x": [1]}),
            ContractCase("empty", ({}, {}), {}),
            ContractCase("fill", ({}, {"x": {"y": 1}}), {"x": {"y": 1}}),
        ),
    ),
    FunctionRepairTask(
        "vare_ig.explore.duration_ms.v1", "exploration",
        "Implement `solve(text)`. Convert a nonnegative decimal duration with suffix `ms`, `s`, `m`, or `h` to integer milliseconds. "
        "Surrounding whitespace is allowed. Return `None` for malformed, negative, unknown-unit, or fractional-millisecond values.",
        (
            ContractCase("seconds", ("2.5s",), 2500),
            ContractCase("millis", ("30ms",), 30),
            ContractCase("minutes", ("1m",), 60000),
            ContractCase("fractional_ms", ("0.1ms",), None),
            ContractCase("bad", ("3weeks",), None),
        ),
    ),
    FunctionRepairTask(
        "vare_ig.explore.relative_path.v1", "exploration",
        "Implement `solve(path)`. Normalize a relative POSIX path by dropping empty and `.` components and resolving `..` against a prior "
        "normal component. Return `None` for absolute paths or attempts to go above the starting directory. Join remaining parts with `/`; "
        "the empty path is `\"\"`.",
        (
            ContractCase("normalize", ("a/./b/../c",), "a/c"),
            ContractCase("collapse", ("a//b",), "a/b"),
            ContractCase("empty", (".",), ""),
            ContractCase("absolute", ("/a",), None),
            ContractCase("escape", ("../../a",), None),
        ),
    ),
)


DEVELOPMENT_TASKS = (
    FunctionRepairTask(
        "vare_ig.dev.parse_duration_pair.v1", "development",
        "Implement `solve(text)`. Parse a nonnegative decimal duration with `ms`, `s`, `m`, or `h` suffix and return whole milliseconds. "
        "Return `None` for invalid or fractional-millisecond input.",
        (
            ContractCase("hour", ("1.25h",), 4500000),
            ContractCase("trim", (" 7s ",), 7000),
            ContractCase("fraction", ("0.5ms",), None),
            ContractCase("negative", ("-1s",), None),
            ContractCase("unit", ("2d",), None),
        ),
    ),
    FunctionRepairTask(
        "vare_ig.dev.lexical_course_order.v1", "development",
        "Implement `solve(nodes, edges)`. Return the lexicographically smallest topological ordering, including isolated nodes. Return `None` "
        "for a cycle or an edge with an unknown endpoint. Duplicate edges do not count twice.",
        (
            ContractCase("order", (["d", "c", "b", "a"], [["c", "a"]]),
                         ["b", "c", "a", "d"]),
            ContractCase("isolated", (["x"], []), ["x"]),
            ContractCase("cycle", (["x", "y"], [["x", "y"], ["y", "x"]]), None),
            ContractCase("duplicate", (["a", "b"], [["a", "b"], ["a", "b"]]), ["a", "b"]),
            ContractCase("unknown", (["a"], [["a", "b"]]), None),
        ),
    ),
    FunctionRepairTask(
        "vare_ig.dev.first_event_ids.v1", "development",
        "Implement `solve(events)`. Each event is `[id, value]`. Keep only the first row for each string id, preserving first-seen order.",
        (
            ContractCase("duplicates", ([["b", 1], ["a", 2], ["b", 3]],), [["b", 1], ["a", 2]]),
            ContractCase("case_sensitive", ([["A", 1], ["a", 2]],), [["A", 1], ["a", 2]]),
            ContractCase("empty", ([],), []),
            ContractCase("all_same", ([["x", 1], ["x", 2]],), [["x", 1]]),
            ContractCase("distinct", ([["c", 1], ["a", 2]],), [["c", 1], ["a", 2]]),
        ),
    ),
    FunctionRepairTask(
        "vare_ig.dev.deep_lookup.v1", "development",
        "Implement `solve(config, path, default)`. Resolve a dot-separated path through nested dictionaries. If any key is missing, "
        "return `default`. An existing value of `None` is a valid result. Empty path returns the entire config. Do not mutate input.",
        (
            ContractCase("nested", ({"a": {"b": 2}}, "a.b", 0), 2),
            ContractCase("missing", ({"a": {}}, "a.b", "fallback"), "fallback"),
            ContractCase("null_exists", ({"a": {"b": None}}, "a.b", 0), None),
            ContractCase("empty_path", ({"x": 1}, "", 0), {"x": 1}),
            ContractCase("empty_config", ({}, "x", None), None),
        ),
    ),
)


FAMILY_BY_TASK = {
    **{task.task_id: "parsing" for task in TRAIN_TASKS if task.task_id.endswith(("ini_sections.v1", "csv_record.v1", "cli_overrides.v1"))},
    **{task.task_id: "graph" for task in TRAIN_TASKS if task.task_id.endswith(("stable_toposort.v1", "graph_reachable.v1", "undirected_components.v1"))},
    **{task.task_id: "stream" for task in TRAIN_TASKS if task.task_id.endswith(("merge_busy_windows.v1", "running_key_totals.v1", "latest_record_by_key.v1"))},
    **{task.task_id: "configuration" for task in TRAIN_TASKS if task.task_id.endswith(("deep_merge_defaults.v1", "normalize_relative_path.v1", "parse_duration_ms.v1"))},
}
FAMILY_BY_EXPLORATION_TASK = {
    task.task_id: family
    for family, suffixes in {
        "parsing": ("parse_csv_header.v1", "ini_overrides.v1"),
        "graph": ("graph_course_order.v1", "shortest_unweighted_path.v1"),
        "stream": ("merge_event_ranges.v1", "running_event_counts.v1", "dedupe_event_ids.v1"),
        "configuration": ("deep_defaults_config.v1", "duration_ms.v1", "relative_path.v1"),
    }.items()
    for task in EXPLORATION_TASKS
    if task.task_id.endswith(suffixes)
}
FAMILY_BY_DEVELOPMENT_TASK = {
    task.task_id: family
    for task, family in zip(DEVELOPMENT_TASKS, ("configuration", "graph", "stream", "configuration"), strict=True)
}


def all_tasks() -> tuple[FunctionRepairTask, ...]:
    return TRAIN_TASKS + EXPLORATION_TASKS + DEVELOPMENT_TASKS


def validate_task_pack() -> None:
    tasks = all_tasks()
    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task IDs")
    if any(len(task.cases) != 5 for task in tasks):
        raise ValueError("VARE-IG tasks must have exactly five grader cases for matched verification work")
    if set(FAMILY_BY_TASK) != {task.task_id for task in TRAIN_TASKS}:
        raise ValueError("every Train task must have one declared family")
    if set(FAMILY_BY_EXPLORATION_TASK) != {task.task_id for task in EXPLORATION_TASKS}:
        raise ValueError("every exploration task must have one declared family")
    if any(task.split != "train" for task in TRAIN_TASKS):
        raise ValueError("Train task split mismatch")
    if any(task.split != "exploration" for task in EXPLORATION_TASKS):
        raise ValueError("exploration task split mismatch")
    if any(task.split != "development" for task in DEVELOPMENT_TASKS):
        raise ValueError("Development task split mismatch")


validate_task_pack()
