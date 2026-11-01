"""Seeded inputs for the rollout edit-apply workload (grader-owned).

The workload is the completion-to-edit step of the learning pipeline: a model
completion is parsed into SEARCH/REPLACE blocks (``parse_blocks``) and applied
to the target file (``apply_blocks``) before any test runs. Each case is one
(original file, completion) pair. Cases cover exact matches, matches that
differ only by trailing whitespace (the relaxed path), missing and ambiguous
SEARCH text, multi-block completions and a ``<think>`` preamble.

This file is executed inside the sandbox before candidate code is imported,
and in the trusted parent to compute expected outputs with the unmodified
module. It must stay standard-library only and deterministic for a seed.
Seeds are drawn by the parent per run and are never visible to a proposer.
"""
import hashlib
import random

# Bound at load time so that later rebinding of hashlib attributes by the code
# under test cannot change the digest computed inside the sandbox.
_sha256 = hashlib.sha256

WORKLOAD_ID = "rollout_edit_apply_v1"
PROFILES = ("benchmark", "differential")

_WORDS = ("self", "value", "items", "result", "config", "batch", "token", "logits",
          "state", "step", "grad", "loss", "x", "y", "idx", "mask", "cache", "layer")


def _code_line(rng, i):
    ind = "    " * rng.randrange(0, 4)
    k = rng.random()
    if k < 0.08:
        return ""
    if k < 0.15:
        return f"{ind}def {rng.choice(_WORDS)}_{i}({rng.choice(_WORDS)}, {rng.choice(_WORDS)}):"
    if k < 0.20:
        return f"{ind}return {rng.choice(_WORDS)}"
    line = f"{ind}{rng.choice(_WORDS)}_{i % 97} = {rng.choice(_WORDS)}.{rng.choice(_WORDS)}({rng.randrange(1000)})"
    if rng.random() < 0.1:
        line += " " * rng.randrange(1, 3)
    return line


def _base_files(rng, n_files, lo, hi):
    return [[_code_line(rng, i) for i in range(rng.randrange(lo, hi))] for _ in range(n_files)]


def _drop_trailing(rng, seg):
    out = []
    for ln in seg:
        if rng.random() < 0.5:
            ln = ln.rstrip() + ("  " if rng.random() < 0.3 else "")
        out.append(ln)
    return out


def _block(search_lines, replace_lines):
    return ("<<<<<<< SEARCH\n" + "\n".join(search_lines) + "\n=======\n"
            + "\n".join(replace_lines) + "\n>>>>>>> REPLACE")


def _case(rng, base, index, profile):
    lines = list(base)
    # A unique marker line keeps every case distinct, so results cannot be reused.
    lines.insert(rng.randrange(len(lines) + 1), f"# case {index} {rng.getrandbits(48):012x}")
    kind = rng.random()
    blocks = []
    edge = profile == "differential"
    if edge and kind < 0.12:
        # Ambiguous after right-stripping: the same window twice, one copy with trailing spaces.
        at = rng.randrange(0, len(lines) - 8)
        seg = lines[at:at + rng.randrange(2, 6)]
        lines[at + len(seg):at + len(seg)] = [ln + "  " for ln in seg]
        blocks.append(_block([ln.rstrip() + "\t" for ln in seg], seg[:-1] + ["pass"]))
    elif edge and kind < 0.20:
        blocks.append(_block([f"not_in_file_{index} = 0"], ["x = 1"]))
    elif edge and kind < 0.26:
        seg = lines[3:6]
        blocks.append(_block(seg, seg))  # no-op edit
    elif edge and kind < 0.30:
        blocks.append(_block(["   "], ["x = 1"]))  # empty search after strip
    elif edge and kind < 0.36:
        # Exact duplicate window: ambiguous on the exact path.
        at = rng.randrange(0, len(lines) - 8)
        seg = lines[at:at + 3]
        lines[at + 3:at + 3] = list(seg)
        blocks.append(_block(seg, ["pass"]))
    else:
        count = rng.randrange(1, 4)
        starts = sorted(rng.sample(range(0, len(lines) - 40), count))
        last = -1
        for s in starts:
            if s <= last:
                continue
            seg = lines[s:s + rng.randrange(3, 15)]
            last = s + len(seg)
            search = _drop_trailing(rng, seg) if rng.random() < 0.6 else list(seg)
            blocks.append(_block(search, seg[:-1] + [seg[-1] + f"  # edited {index}"]))
    text = "\n".join(lines) + "\n"
    head = "<think>plan: edit the file</think>\n" if rng.random() < 0.2 else ""
    return text, head + "Here is the fix.\n" + "\n\n".join(blocks)


def make_cases(seed, count, profile="benchmark"):
    """``count`` distinct (original, completion) pairs for ``seed``."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}")
    rng = random.Random(seed)
    bases = _base_files(rng, 6, 800, 2500)
    return [_case(rng, bases[rng.randrange(len(bases))], i, profile) for i in range(count)]


def outcome_digest(outcomes):
    """sha256 over (ok, reason, sha256(new_text)) for every case, in order."""
    h = _sha256()
    for ok, reason, new_text in outcomes:
        text_digest = "-" if new_text is None else _sha256(new_text.encode()).hexdigest()
        h.update(f"{int(ok)}\x1f{reason}\x1f{text_digest}\x1e".encode())
    return h.hexdigest()


def run_cases(module, cases):
    """Reference execution used by the trusted parent (unmodified module)."""
    out = []
    for original, completion in cases:
        o = module.apply_blocks(original, module.parse_blocks(completion))
        out.append((o.ok, o.reason, o.new_text))
    return out
