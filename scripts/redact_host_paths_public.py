#!/usr/bin/env python3
"""Regenerate and verify redacted public copies of committed files that record host paths.

Many historical result, protocol and documentation files record absolute paths of the hosts
that produced them: user home directories, per-session working directories, per-user temporary
directories and local tool prefixes. The originals are hash-bound evidence and stay unchanged.
This script derives public copies under ``results/public-redacted/<original repository-relative
path>`` by applying a fixed, ordered list of path substitutions, and records every original and
copy in ``results/public-redacted/MAPPING.json`` and in the table in
``results/public-redacted/README.md``.

It is the repository-wide sibling of ``scripts/redact_learning_pipeline_v1_public.py``. Files
already covered by that script's ``MAPPING.json`` are listed as excluded here rather than copied
a second time.

The set of copied files is derived, not hand-listed: every tracked UTF-8 file in which a
substitution applies gets a copy unless an exclusion rule below names it. ``--check`` then fails
if

* an original no longer has the sha256 recorded in ``MAPPING.json``, or a new file with a host
  path has appeared (the regenerated mapping differs from the committed one);
* a copy or the README table differs from its regenerated form;
* a JSON / JSON Lines copy is invalid or is not the original structure with redacted strings;
* a copy still contains a host path or a JUnit XML testsuite ``hostname`` value other than the
  placeholder, or a file without a copy contains one and is not excluded.

Usage:
  python scripts/redact_host_paths_public.py --write   # regenerate copies, MAPPING.json, README table
  python scripts/redact_host_paths_public.py --check   # verify (exit 1 on any problem)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEST = Path("results/public-redacted")
MAPPING = DEST / "MAPPING.json"
README = DEST / "README.md"
SCHEMA = "vare_public_redaction_repository_v1"
SIBLING_DEST = Path("results/learning-pipeline-v1/public-redacted")
SIBLING_MAPPING = SIBLING_DEST / "MAPPING.json"
# A copy at or above this size would be a new large blob and fail scripts/check_tracked_blob_budget.py.
MAX_COPY_BYTES = 1024 * 1024
TABLE_BEGIN = "<!-- BEGIN GENERATED MAPPING TABLE -->"
TABLE_END = "<!-- END GENERATED MAPPING TABLE -->"

# One path component. Quotes, backticks, escapes, markup and separators end a component so that
# paths inside Python reprs, JSON strings, ANSI-coloured logs and XML-escaped text are bounded.
_SEG = r"[^/\s\"'`\\\x1b<>&|,;:()\[\]{}]+"
_NODOT = r"[^/\s\"'`\\\x1b<>&|,;:()\[\]{}.]*"

# Ordered regular-expression substitutions; each rule runs on the output of the previous one.
# Path components are matched generically so that no host-specific directory name is spelled here.
RULES: tuple[tuple[str, str], ...] = (
    # Session directory cut short by a log formatter ("...", "…"); the ellipsis itself is kept.
    (rf"/Users/{_SEG}/Documents/{_SEG}/(?:{_NODOT}/)?{_NODOT}(?=\.\.\.|…)", "<SESSION_ROOT>"),
    (rf"/Users/{_SEG}/Documents/{_SEG}/{_SEG}/{_SEG}", "<SESSION_ROOT>"),
    (rf"/workspace/scratch/{_SEG}", "<SESSION_ROOT>"),
    (rf"/Users/{_SEG}/\.cache/huggingface/hub", "<HF_HUB_CACHE>"),
    (rf"/Users/(?!runner/){_SEG}", "<HOME>"),
    (rf"(?:/private)?/var/folders/{_SEG}/{_SEG}/T\b", "<TMPDIR>"),
    (rf"(?:/private)?/var/folders/{_SEG}/{_NODOT}(?=\.\.\.|…)", "<TMPDIR>"),
    (r"/private/tmp\b", "<TMP>"),
    (r"/opt/homebrew\b", "<HOMEBREW_PREFIX>"),
    (rf"/mnt/{_SEG}/VARE\b", "<REPO_ROOT>"),
    (rf"/workspace/{_SEG}/lp-cache", "<DATA_CACHE>"),
)
PLACEHOLDERS = {
    "<SESSION_ROOT>": "per-session working directory on a development host",
    "<HF_HUB_CACHE>": "Hugging Face hub cache directory in the user's home directory",
    "<HOME>": "user home directory on a macOS development host",
    "<TMPDIR>": "per-user temporary directory on a macOS host ($TMPDIR)",
    "<TMP>": "system temporary directory on a macOS host (/tmp)",
    "<HOMEBREW_PREFIX>": "Homebrew installation prefix on a macOS host",
    "<REPO_ROOT>": "repository checkout on a mounted data volume",
    "<DATA_CACHE>": "dataset cache directory on the data-preparation host",
}

# Machine hostname recorded by JUnit XML reporters on <testsuite>/<testsuites> elements. It is
# host-specific metadata that is not needed to reproduce a result, so its value is replaced.
HOSTNAME_PLACEHOLDER = "redacted-host"
JUNIT_HOSTNAME = re.compile(r'(<testsuites?\b[^>]*?\shostname=)(["\'])((?:(?!\2)[^<>])*)\2')
JUNIT_HOSTNAME_MEANING = "machine hostname in a JUnit XML testsuite element"


def redact_junit_hostnames(text: str) -> tuple[str, int]:
    """Replace JUnit XML ``hostname`` attribute values that are not already the placeholder."""
    n = 0

    def sub(m: re.Match[str]) -> str:
        nonlocal n
        if m.group(3) == HOSTNAME_PLACEHOLDER:
            return m.group(0)
        n += 1
        return f"{m.group(1)}{m.group(2)}{HOSTNAME_PLACEHOLDER}{m.group(2)}"

    return JUNIT_HOSTNAME.sub(sub, text), n


def junit_hostname_hits(text: str) -> list[str]:
    return sorted({m.group(3) for m in JUNIT_HOSTNAME.finditer(text) if m.group(3) != HOSTNAME_PLACEHOLDER})


# Anything that looks like an absolute host path. The look-behind skips relative paths such as
# ``task/workspace/public`` or ``../work/private/x``.
HOST_PATH = re.compile(
    r"(?<![\w.~-])(?:/Users/|/home/|/private/|/var/folders/|/Volumes/|/root/|/workspace/"
    r"|/opt/homebrew/|/mnt/|[A-Za-z]:\\\\Users\\\\|[A-Za-z]:\\Users\\)"
)
# Absolute paths that match HOST_PATH but do not describe a host of this project. They are kept
# verbatim in copies and do not on their own require a copy.
NON_HOST: tuple[tuple[str, str], ...] = (
    (r"/workspace/(?:src\b|[\w.-]+\.(?:py|json|txt)\b)", "file at the evaluation-container mount point"),
    (r"/home/runner/work/", "GitHub-hosted Linux runner checkout"),
    (r"/Users/runner/", "upstream build path on a GitHub-hosted macOS runner"),
    (r"/home/conda/feedstock_root/", "upstream conda-forge build path"),
    (r"/root/(?=[a-z_]+\b)(?![\w.]*/)", "reviewer agent identifier, not a file path"),
    (r"/home/user/src/", "path written by a model inside its sandboxed task workspace"),
    (r"/private/var/folders(?:/\.\.\.)?(?![\w/-])", "generic macOS temporary-directory root named in prose"),
)
_NON_HOST = tuple(re.compile(p) for p, _ in NON_HOST)

# Files that contain host paths but get no copy here, with the reason.
EXCLUDE: tuple[tuple[str, str], ...] = (
    (r"^scripts/redact_[\w]+\.py$|^tests/test_redact_[\w]+\.py$",
     "redaction script or its test; the paths are patterns and synthetic fixtures"),
    (r"^(?:scripts|tests|src|benchmarks)/",
     "executable source; a copy with placeholders would not run, so a host literal here needs a "
     "source change rather than a derived copy"),
)


# Files withheld from public copies until their third-party repository references have been
# reviewed separately. Listed by explicit path so that no selection pattern is committed.
PENDING_REVIEW_REASON = "contains third-party repository references pending separate review"
PENDING_REVIEW: frozenset[str] = frozenset({
    "research/flagship/DECISIONS.md",
    "results/grpo-audit-independent-reproduction-v2/run-1/attempt-1-failure.log",
    "results/isolation-boundary-verification-v1/run-1-colima-m1/observations.json",
    "results/isolation-boundary-verification-v1/run-1-colima-m1/run-progress.json",
    "results/isolation-boundary-verification-v1/run-2-colima-m1/observations.json",
    "results/isolation-boundary-verification-v1/run-2-colima-m1/run-progress.json",
    "results/isolation-boundary-verification-v1/run-3-colima-m1/observations.json",
    "results/isolation-boundary-verification-v1/run-3-colima-m1/run-progress.json",
    "results/promotion-cancellation-ownership-v1/run-1/baseline-pytest.log",
    "results/promotion-cancellation-ownership-v1/run-1/candidate-pytest.log",
    "results/rvl-inference-transaction-v1/pr58-freshness-compat/alias_only-pytest.log",
    "results/rvl-inference-transaction-v1/pr58-freshness-compat/baseline-pytest.log",
    "results/rvl-inference-transaction-v1/pr58-freshness-compat/fixed-pytest.log",
    "results/rvl-inference-transaction-v1/pr58-freshness-compat/integration_main-pytest.log",
    "results/trusted-scoring-boundary-v1/run-2-colima-m1/observations.json",
    "results/trusted-scoring-boundary-v1/run-3-colima-m1/observations.json",
    "results/trusted-scoring-boundary-v1/run-4-colima-m1/observations.json",
    "results/trusted-scoring-boundary-v1/run-5-colima-m1/observations.json",
    "results/trusted-scoring-boundary-v1/run-6-colima-m1/observations.json",
})


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def redact_text(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for pattern, placeholder in RULES:
        text, n = re.subn(pattern, placeholder, text)
        if n:
            counts[placeholder] = counts.get(placeholder, 0) + n
    text, n = redact_junit_hostnames(text)
    if n:
        counts[HOSTNAME_PLACEHOLDER] = n
    return text, counts


def redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)[0]
    if isinstance(value, list):
        return [redact_value(v) for v in value]
    if isinstance(value, dict):
        return {redact_text(k)[0]: redact_value(v) for k, v in value.items()}
    return value


def parse(path: str, text: str) -> list[Any]:
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return [json.loads(text)]


def _parses(path: str, text: str) -> bool:
    try:
        parse(path, text)
    except json.JSONDecodeError:
        return False
    return True


def _non_host_reason(text: str, pos: int) -> str | None:
    for (_, reason), rx in zip(NON_HOST, _NON_HOST):
        if rx.match(text, pos):
            return reason
    return None


def host_path_hits(text: str) -> list[str]:
    """Distinct host-path prefixes in ``text`` that are not an allowed non-host form."""
    return sorted({m.group(0) for m in HOST_PATH.finditer(text) if _non_host_reason(text, m.start()) is None})


def non_host_forms(text: str) -> list[str]:
    return sorted({r for m in HOST_PATH.finditer(text) if (r := _non_host_reason(text, m.start()))})


_MARKERS = (b"/Users/", b"/home/", b"/private/", b"/var/folders/", b"/Volumes/", b"/root/", b"/workspace/",
            b"/opt/homebrew/", b"/mnt/", b":\\Users\\", b":\\\\Users\\\\")


def candidate_files(root: Path) -> list[str]:
    if (root / ".git").exists():
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], check=True, capture_output=True).stdout
        rels = [r for r in out.decode("utf-8").split("\0") if r]
    else:
        rels = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]
    skip = (DEST.as_posix() + "/", SIBLING_DEST.as_posix() + "/")
    return sorted(r for r in rels if not r.startswith(skip) and (root / r).is_file())


def _sibling_covered(root: Path) -> dict[str, str]:
    path = root / SIBLING_MAPPING
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    covered = {f["original"]: f["original_sha256"] for f in data["files"]}
    covered[SIBLING_MAPPING.as_posix()] = ""
    return covered


def build(root: Path = ROOT) -> tuple[dict[str, Any], dict[Path, bytes], list[str]]:
    problems: list[str] = []
    copies: dict[Path, bytes] = {}
    files: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    sibling = _sibling_covered(root)
    for rel in candidate_files(root):
        raw = (root / rel).read_bytes()
        if not any(m in raw for m in _MARKERS):
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            excluded.append({"path": rel, "sha256": sha256(raw), "reason": "not UTF-8 text"})
            continue
        red, counts = redact_text(text)
        remaining = host_path_hits(red)
        if not counts and not remaining:
            forms = non_host_forms(text)
            if forms:
                excluded.append({"path": rel, "sha256": sha256(raw),
                                 "reason": "only non-host path forms: " + "; ".join(forms)})
            continue
        if rel in sibling:
            if sibling[rel] and sibling[rel] != sha256(raw):
                problems.append(f"{rel}: sha256 differs from {SIBLING_MAPPING.as_posix()}")
            reason = (f"redaction rule patterns of {SIBLING_MAPPING.as_posix()}" if not sibling[rel]
                      else f"copy maintained in {SIBLING_DEST.as_posix()}/")
            excluded.append({"path": rel, "sha256": sha256(raw), "reason": reason})
            continue
        rule = next((reason for pat, reason in EXCLUDE if re.search(pat, rel)), None)
        if rel in PENDING_REVIEW:
            rule = PENDING_REVIEW_REASON
        if rule:
            excluded.append({"path": rel, "sha256": sha256(raw), "reason": rule})
            continue
        data = red.encode("utf-8")
        if len(data) >= MAX_COPY_BYTES:
            excluded.append({"path": rel, "sha256": sha256(raw),
                             "reason": f"copy would be {len(data)} bytes, over the {MAX_COPY_BYTES}-byte "
                                       "tracked-blob budget; host paths: " + ", ".join(sorted(counts))})
            continue
        if remaining:
            problems.append(f"{rel}: host path remains after redaction: {remaining}")
        # A ``.json`` file that is not JSON (for example captured stdout with a trailing line) is
        # copied as text; every file that parses must still parse to the same redacted structure.
        if rel.endswith((".json", ".jsonl")) and _parses(rel, text):
            try:
                if parse(rel, red) != [redact_value(v) for v in parse(rel, text)]:
                    problems.append(f"{rel}: redacted copy is not the original structure with redacted strings")
            except json.JSONDecodeError as exc:
                problems.append(f"{rel}: redacted copy is not valid JSON: {exc}")
        out_rel = DEST / rel
        copies[out_rel] = data
        files.append({"original": rel, "original_sha256": sha256(raw), "original_bytes": len(raw),
                      "redacted": out_rel.as_posix(), "redacted_sha256": sha256(data),
                      "redacted_bytes": len(data), "replacements": dict(sorted(counts.items()))})
    mapping = {
        "schema": SCHEMA,
        "rule": {"type": "ordered regular-expression substitution (Python re) on the UTF-8 file text; "
                         "each rule is applied to the output of the previous one",
                 "substitutions": [{"pattern": p, "placeholder": ph} for p, ph in RULES],
                 "placeholders": PLACEHOLDERS,
                 "verification_pattern": HOST_PATH.pattern,
                 "junit_hostname": {"pattern": JUNIT_HOSTNAME.pattern, "value_group": 3,
                                    "placeholder": HOSTNAME_PLACEHOLDER, "meaning": JUNIT_HOSTNAME_MEANING},
                 "non_host_forms": [{"pattern": p, "meaning": m} for p, m in NON_HOST]},
        "totals": {"copied_files": len(files),
                   "replacements": sum(sum(f["replacements"].values()) for f in files),
                   "excluded_files": len(excluded)},
        "files": files,
        "excluded": excluded,
    }
    return mapping, copies, problems


def _cell(counts: dict[str, int]) -> str:
    return ", ".join(f"{k} x{v}" for k, v in counts.items())


def render_table(mapping: dict[str, Any]) -> str:
    t = mapping["totals"]
    lines = [TABLE_BEGIN, "",
             f"{t['copied_files']} copies, {t['replacements']} replacements, {t['excluded_files']} excluded files.",
             "", "### Copies", "",
             "| Original | Original sha256 | Redacted sha256 | Replacements |", "|---|---|---|---|"]
    for f in mapping["files"]:
        lines.append(f"| `{f['original']}` | `{f['original_sha256']}` | `{f['redacted_sha256']}` | "
                     f"{_cell(f['replacements'])} |")
    lines += ["", "### Excluded", "", "| File | sha256 | Reason |", "|---|---|---|"]
    for e in mapping["excluded"]:
        lines.append(f"| `{e['path']}` | `{e['sha256']}` | {e['reason']} |")
    lines += ["", TABLE_END]
    return "\n".join(lines)


def render_readme(current: str, mapping: dict[str, Any]) -> str | None:
    start, end = current.find(TABLE_BEGIN), current.find(TABLE_END)
    if start < 0 or end < start:
        return None
    return current[:start] + render_table(mapping) + current[end + len(TABLE_END):]


def _committed_text_hits(root: Path) -> list[str]:
    problems = []
    for p in sorted((root / DEST).rglob("*")):
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8")
        if p == root / MAPPING:  # the rule block holds the patterns themselves, not host paths
            text = json.dumps({k: v for k, v in json.loads(text).items() if k != "rule"})
        elif p == root / README:  # the README names the replaced prefixes in its rule table
            text = text[text.find(TABLE_BEGIN):]
        hits = host_path_hits(text)
        if hits:
            problems.append(f"{p.relative_to(root).as_posix()}: host path present: {hits}")
        names = junit_hostname_hits(text)
        if names:
            problems.append(f"{p.relative_to(root).as_posix()}: JUnit hostname present: {len(names)} value(s)")
    return problems


def check(root: Path = ROOT) -> list[str]:
    mapping, copies, problems = build(root)
    committed_path = root / MAPPING
    if not committed_path.exists():
        return [*problems, f"{MAPPING.as_posix()} missing"]
    committed = json.loads(committed_path.read_text())
    if committed != mapping:
        old = {f["original"]: f["original_sha256"] for f in committed.get("files", [])}
        new = {f["original"]: f["original_sha256"] for f in mapping["files"]}
        for rel in sorted(old.keys() | new.keys()):
            if rel not in new:
                problems.append(f"{rel}: listed in {MAPPING.as_posix()} but no longer needs a copy")
            elif rel not in old:
                problems.append(f"{rel}: contains a host path but has no redacted copy")
            elif old[rel] != new[rel]:
                problems.append(f"{rel}: original sha256 differs from {MAPPING.as_posix()}")
        problems.append(f"{MAPPING.as_posix()} differs from the regenerated mapping "
                        "(an original changed, a host path appeared or the rule changed)")
    expected = {(root / rel) for rel in copies} | {root / MAPPING, root / README}
    for rel, data in copies.items():
        p = root / rel
        if not p.exists() or p.read_bytes() != data:
            problems.append(f"{rel.as_posix()}: committed copy differs from the regenerated copy")
    for p in sorted((root / DEST).rglob("*")):
        if p.is_file() and p not in expected:
            problems.append(f"{p.relative_to(root).as_posix()}: file in {DEST.as_posix()} has no mapping entry")
    readme = root / README
    if not readme.exists() or render_readme(readme.read_text(encoding="utf-8"), mapping) != readme.read_text(
            encoding="utf-8"):
        problems.append(f"{README.as_posix()}: generated mapping table is missing or stale")
    return problems + _committed_text_hits(root)


def write(root: Path = ROOT) -> list[str]:
    mapping, copies, problems = build(root)
    readme = root / README
    current = readme.read_text(encoding="utf-8") if readme.exists() else f"{TABLE_BEGIN}\n{TABLE_END}\n"
    rendered = render_readme(current, mapping)
    if rendered is None:
        problems.append(f"{README.as_posix()}: table markers missing")
    if problems:
        return problems
    for p in sorted((root / DEST).rglob("*"), reverse=True):
        if p.is_file() and p not in {root / MAPPING, readme}:
            p.unlink()
    for rel, data in copies.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    (root / MAPPING).write_text(json.dumps(mapping, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                                encoding="utf-8")
    assert rendered is not None
    readme.write_text(rendered, encoding="utf-8")
    for p in sorted((root / DEST).rglob("*"), reverse=True):
        if p.is_dir() and not any(p.iterdir()):
            p.rmdir()
    return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    problems = write() if args.write else check()
    for p in problems:
        print(f"FAIL {p}", file=sys.stderr)
    if not problems:
        mapping = json.loads((ROOT / MAPPING).read_text())
        t = mapping["totals"]
        print(f"OK {t['copied_files']} redacted copies ({t['replacements']} replacements) verified; "
              f"{t['excluded_files']} files excluded")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
