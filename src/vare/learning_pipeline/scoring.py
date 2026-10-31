"""Candidate scoring with two recorded verifiers.

* ``cheap_f2p_v1``   - runs only the FAIL_TO_PASS tests (fast, can accept a
  candidate that breaks other behavior: a false positive source).
* ``trusted_full_v1`` - FAIL_TO_PASS plus every PASS_TO_PASS test. This is the
  reward used for training-data admission and the only metric used for
  held-out claims.

Both verdicts and their costs are logged for every candidate so later
studies of budgeted verifier-error correction and of allocating trusted
verification during training can be replayed offline from full-information
logs without re-running the policy. v1 itself always runs both.
"""
from __future__ import annotations

import difflib
import hashlib
from typing import Any, Mapping

from .edit_format import apply_blocks, parse_blocks

CHEAP = "cheap_f2p_v1"
TRUSTED = "trusted_full_v1"


def score_candidate(env, task: Mapping[str, Any], buggy_text: str, output_text: str,
                    *, timeout: float = 180) -> dict[str, Any]:
    blocks = parse_blocks(output_text)
    applied = apply_blocks(buggy_text, blocks)
    rec: dict[str, Any] = {
        "instance_id": task["instance_id"],
        "output_sha256": hashlib.sha256(output_text.encode()).hexdigest(),
        "n_blocks": len(blocks),
        "apply": applied.reason,
    }
    if not applied.ok:
        rec.update(outcome="malformed_or_unapplied_edit", reward=0,
                   verifiers={v: {"passed": False, "reason": applied.reason, "seconds": 0.0,
                                  "executed": False, "tests": {}}
                              for v in (CHEAP, TRUSTED)})
        return rec
    rec["candidate_sha256"] = hashlib.sha256(applied.new_text.encode()).hexdigest()
    rec["final_diff"] = "".join(difflib.unified_diff(
        buggy_text.splitlines(keepends=True), applied.new_text.splitlines(keepends=True),
        fromfile=f"a/{task['target_file']}", tofile=f"b/{task['target_file']}"))
    common = dict(bug_patch=task["_patch"], target_file=task["target_file"],
                  candidate_text=applied.new_text, timeout=timeout)
    cheap = env.grade(fail_to_pass=task["fail_to_pass"], pass_to_pass=[], **common)
    trusted = env.grade(fail_to_pass=task["fail_to_pass"], pass_to_pass=task["pass_to_pass"], **common)
    rec["verifiers"] = {CHEAP: cheap, TRUSTED: trusted}
    rec["reward"] = int(bool(trusted["passed"]))
    rec["outcome"] = "pass" if trusted["passed"] else "edit_applied_but_wrong"
    rec["cheap_false_positive"] = bool(cheap["passed"] and not trusted["passed"])
    return rec


def calibrate(env, task: Mapping[str, Any], *, timeout: float = 180) -> dict[str, Any]:
    """Baseline (bug, no fix) must fail F2P and pass P2P; reference fix must pass all."""
    from .edit_format import reference_fix_blocks

    base = env.grade(bug_patch=task["_patch"], target_file=task["target_file"], candidate_text=None,
                     fail_to_pass=task["fail_to_pass"], pass_to_pass=task["pass_to_pass"], timeout=timeout)
    buggy = env.buggy_file_text(task["_patch"], task["target_file"])
    _, blocks = reference_fix_blocks(task["_patch"])
    fixed = apply_blocks(buggy, blocks)
    ref = env.grade(bug_patch=task["_patch"], target_file=task["target_file"],
                    candidate_text=fixed.new_text if fixed.ok else buggy,
                    fail_to_pass=task["fail_to_pass"], pass_to_pass=task["pass_to_pass"], timeout=timeout)
    baseline_fails = (base.get("f2p_passed", 0) < base.get("f2p_total", 1)) and base.get("reason") != "bug_patch_apply_error"
    baseline_p2p_ok = base.get("p2p_passed") == base.get("p2p_total")
    qualified = bool(baseline_fails and baseline_p2p_ok and fixed.ok and ref["passed"])
    return {
        "instance_id": task["instance_id"],
        "qualified": qualified,
        "baseline_fails_f2p": bool(baseline_fails),
        "baseline_passes_p2p": bool(baseline_p2p_ok),
        "reference_passes": bool(ref["passed"]),
        "baseline": base,
        "reference": ref,
    }
