"""Run-002 held-out pool: candidate repositories, licence allowlist extension, seeded order.

Shared by ``scripts/run002_build_check.py`` and ``scripts/run002_freeze_heldout.py``.

The run-001 package ``src/vare/learning_pipeline`` (including ``sources.REPO_LICENSES``) is
hash-frozen by ``protocols/coding_policy_learning_mlx_4b_v1.json`` and is enumerated by the v4
continuation source inventory, so the allowlist extension for run-002 lives here instead of
editing the frozen module. ``admitted_licenses()`` is the run-002 admission table: the frozen
v1 allowlist plus ``RUN002_REPO_LICENSES``. Neither run-001 selection nor the v1 builder
changes, because the v1 builder only reads the repositories configured in
``configs/learning_pipeline_v1/curriculum.json``.

Inputs are the byte-pinned files in ``configs/learning_pipeline_run002`` (IDs, counts and hashes
only). Nothing in this module reads task content.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vare.learning_pipeline import curriculum, exclusions, sources  # noqa: E402

INPUT_DIR = ROOT / "configs" / "learning_pipeline_run002"
V1_CONFIG_DIR = ROOT / "configs" / "learning_pipeline_v1"
SEED = "vare-run002-heldout-pool-v1"

CRITERIA_FILE = "allowlist_extension_criteria.md"
COUNTS_FILE = "content_filter_counts.json"
PASSERS_FILE = "heldout_pool_passer_ids.json"
LEDGER_FILE = "exposure_ledger_ids.txt"

# Pinned before any run-002 environment build. Must equal configs/learning_pipeline_run002/SHA256SUMS.
INPUT_SHA256: dict[str, str] = {
    CRITERIA_FILE: "7ed490f8945f9ad395d3f5752540aa5e5148aca6cd99c6d541e3f6ba8e8f3cab",
    COUNTS_FILE: "fde38c27149dec40367dcbaa14e5da2d28fe270b92592e2b6ace77b57ba1fb3c",
    "content_filter_counts.md": "8de1e0ae4b3c5c6dff171a6bbdcb571dd11ea1c9fb77ad511fbc69569c27ad35",
    PASSERS_FILE: "641e6cc366f8408e136d29557a2a8b7a44b7db517eba017546429970c3a7774e",
    LEDGER_FILE: "3ad576efcbf38d43673dd42854f887ed66794a44416f63d2c55d26cc5494f480",
}
CRITERIA_SHA256 = INPUT_SHA256[CRITERIA_FILE]

# Group B of the frozen criteria (section 1) that pass R1-R5 and R6 for k in {7, 8}:
# licence SPDX id and the survey source it came from. Every entry was re-checked with the
# code-host licence API (``license_api_recheck`` in content_filter_counts.json) and returned
# the same permissive SPDX id; ``rebench_v2_field`` entries are admitted only because of
# that re-check (criterion R1).
RUN002_REPO_LICENSES: dict[str, str] = {
    "alanjds/drf-nested-routers": "Apache-2.0",
    "cantools/cantools": "MIT",
    "conan-io/conan": "MIT",
    "cookiecutter/cookiecutter": "BSD-3-Clause",
    "django/channels": "BSD-3-Clause",
    "facebookresearch/fvcore": "Apache-2.0",
    "facelessuser/soupsieve": "MIT",
    "getmoto/moto": "Apache-2.0",
    "iterative/dvc": "Apache-2.0",
    "john-kurkowski/tldextract": "BSD-3-Clause",
    "jsvine/pdfplumber": "MIT",
    "mido/mido": "MIT",
    "pdfminer/pdfminer.six": "MIT",
    "pndurette/gTTS": "MIT",
    "pudo/dataset": "MIT",
    "rsalmei/alive-progress": "MIT",
    "rubik/radon": "MIT",
    "scrapy/scrapy": "BSD-3-Clause",
    "seatgeek/thefuzz": "MIT",
    "sqlfluff/sqlfluff": "MIT",
    "stanfordnlp/dspy": "MIT",
    "tkrajina/gpxpy": "Apache-2.0",
    "tobymao/sqlglot": "MIT",
    "tox-dev/pipdeptree": "MIT",
    "tweepy/tweepy": "MIT",
    "weaveworks/grafanalib": "Apache-2.0",
}
RUN002_LICENSE_SOURCES: dict[str, str] = {
    "alanjds/drf-nested-routers": "github_api",
    "cantools/cantools": "github_api",
    "conan-io/conan": "rebench_v2_field",
    "cookiecutter/cookiecutter": "github_api",
    "django/channels": "rebench_v2_field",
    "facebookresearch/fvcore": "github_api",
    "facelessuser/soupsieve": "rebench_v2_field",
    "getmoto/moto": "rebench_v2_field",
    "iterative/dvc": "rebench_v2_field",
    "john-kurkowski/tldextract": "github_api",
    "jsvine/pdfplumber": "github_api",
    "mido/mido": "github_api",
    "pdfminer/pdfminer.six": "rebench_v2_field",
    "pndurette/gTTS": "github_api",
    "pudo/dataset": "github_api",
    "rsalmei/alive-progress": "github_api",
    "rubik/radon": "github_api",
    "scrapy/scrapy": "rebench_v2_field",
    "seatgeek/thefuzz": "github_api",
    "sqlfluff/sqlfluff": "github_api",
    "stanfordnlp/dspy": "rebench_v2_field",
    "tkrajina/gpxpy": "github_api",
    "tobymao/sqlglot": "github_api",
    "tox-dev/pipdeptree": "rebench_v2_field",
    "tweepy/tweepy": "github_api",
    "weaveworks/grafanalib": "github_api",
}

# Design bounds. Criterion R6 allows k in {7, 8}; the frozen design uses k = 8 only (the k = 7
# pool size is kept for the pool-count evidence). The cap R <= 32 is fixed before any build
# check; the 26 x 8 refusal floor is FLOOR_REPOS in scripts/run002_freeze_heldout.py.
MAX_REPOS = 32
MAX_TASKS_PER_REPO = 8
MIN_TASKS_PER_REPO = 7


@dataclass(frozen=True)
class Candidate:
    upstream: str
    group: str  # "A" (frozen v1 allowlist) or "B" (RUN002_REPO_LICENSES)
    mirror: str
    commit: str
    license: str
    license_source: str
    passers: tuple[str, ...]  # content-filter passers in ascending task order key

    @property
    def repo_key(self) -> str:
        return repo_order_key(self.upstream)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return curriculum.sha256_file(path)


def sha256_json(obj: Any) -> str:
    """Hash convention of the v1 manifests (``sha256(json.dumps(list))``)."""
    return curriculum.sha256_text(json.dumps(obj))


def task_order_key(instance_id: str) -> str:
    return curriculum.order_key(SEED, instance_id)


def repo_order_key(upstream: str) -> str:
    return curriculum.sha256_text(f"{SEED}:repo:{upstream}")


def admitted_licenses() -> dict[str, str]:
    """Frozen v1 allowlist plus the run-002 extension (no overrides allowed)."""
    overlap = set(sources.REPO_LICENSES) & set(RUN002_REPO_LICENSES)
    if overlap:
        raise ValueError(f"run-002 extension overrides v1 entries: {sorted(overlap)}")
    return {**sources.REPO_LICENSES, **RUN002_REPO_LICENSES}


def run001_repositories() -> tuple[list[str], list[str]]:
    cfg = json.loads((V1_CONFIG_DIR / "curriculum.json").read_text())
    return list(cfg["train_repositories"]), list(cfg["heldout_repositories"])


def verify_inputs(input_dir: Path = INPUT_DIR) -> dict[str, str]:
    """Return {file: sha256}; raise if any pinned input differs."""
    got = {name: sha256_file(input_dir / name) for name in INPUT_SHA256}
    bad = {k: v for k, v in got.items() if v != INPUT_SHA256[k]}
    if bad:
        raise RuntimeError(f"run-002 input hash mismatch: {sorted(bad)}")
    return got


def load_passers(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text())
    if data.get("seed") != SEED:
        raise RuntimeError("passer file seed differs from the frozen seed")
    return {repo: list(ids) for repo, ids in data["repos"].items()}


def load_candidates(input_dir: Path = INPUT_DIR, *, min_passers: int = MIN_TASKS_PER_REPO,
                    passers_path: Path | None = None) -> list[Candidate]:
    """Repos admissible before the environment stage (R1-R6) with >= ``min_passers``
    passers, sorted by the seeded repository order key."""
    counts = json.loads((input_dir / COUNTS_FILE).read_text())
    if counts["criteria_sha256"] != CRITERIA_SHA256:
        raise RuntimeError("content-filter counts were not produced under the frozen criteria")
    passers = load_passers(passers_path or input_dir / PASSERS_FILE)
    licences = admitted_licenses()
    train, held = run001_repositories()
    out = []
    for upstream, rec in counts["repos"].items():
        ids = passers.get(upstream, [])
        if not rec["admissible_pre_env"] or len(ids) < min_passers:
            continue
        if upstream in train or upstream in held:
            raise RuntimeError(f"run-001 repository in the candidate pool: {upstream}")
        lic = licences.get(upstream)
        if lic is None or not sources.is_permissive(lic):
            raise RuntimeError(f"{upstream}: not in the run-002 admission table")
        if lic != rec["license_api_recheck"]:
            raise RuntimeError(f"{upstream}: licence {lic} differs from the API re-check")
        if sources.upstream_repo(rec["mirror"])[0] != upstream:
            raise RuntimeError(f"{upstream}: mirror {rec['mirror']} does not match")
        keys = [task_order_key(i) for i in ids]
        if keys != sorted(keys):
            raise RuntimeError(f"{upstream}: passer IDs are not in seeded order")
        out.append(Candidate(upstream=upstream, group=rec["group"], mirror=rec["mirror"],
                             commit=rec["mirror_commit"], license=lic,
                             license_source=rec["license_survey"][0], passers=tuple(ids)))
    return sorted(out, key=lambda c: c.repo_key)


def read_ids(path: Path) -> set[str]:
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip()}


def manifest_ids(path: Path) -> set[str]:
    return {t["instance_id"] for t in curriculum.load_manifest(path)}


def exposed_ids(input_dir: Path = INPUT_DIR) -> dict[str, set[str]]:
    return {
        "train_v1": manifest_ids(V1_CONFIG_DIR / "train.manifest.jsonl"),
        "heldout_run001": manifest_ids(V1_CONFIG_DIR / "heldout.manifest.jsonl"),
        "exposure_ledger": read_ids(input_dir / LEDGER_FILE),
    }


def exclusion_scan(vare_root: Path = ROOT) -> dict[str, Any]:
    """The v1 builder's repository-level exclusion scan (criterion R2)."""
    cfg = json.loads((V1_CONFIG_DIR / "curriculum.json").read_text())
    scan = exclusions.scan_vare_tree(vare_root, skip=cfg.get("scan_skip", ()))
    return {"scan": scan, "excluded": exclusions.excluded_repositories(scan),
            "required_missing": exclusions.check_required(scan)}


def relativize(text: str, roots: Iterable[tuple[Path | str, str]] = ()) -> str:
    """Replace host paths with stable labels so reports carry no absolute host path."""
    pairs = [(str(p), label) for p, label in roots]
    pairs += [(str(ROOT), "<vare>"), (str(Path.home()), "~"), (sys.prefix, "<python-prefix>")]
    for path, label in sorted(pairs, key=lambda x: -len(x[0])):
        if path and path not in ("/", "."):
            text = text.replace(path, label)
    return text


def rel_label(path: Path) -> str:
    """Path relative to the VARE root or the home directory, never absolute."""
    p = Path(os.path.abspath(path))
    for base, label in ((ROOT, ""), (Path.home(), "~/")):
        try:
            rel = p.relative_to(base).as_posix()
        except ValueError:
            continue
        return f"{label}{rel}" if label else rel
    return f"<external>/{p.name}"


def atomic_write_json(path: Path, obj: Mapping[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
