"""Pinned task sources and license records.

Each source is pinned to an exact Hugging Face dataset revision. The dataset
license and the license of every upstream repository that contributes tasks are
recorded in the curriculum manifest. Only permissively licensed repositories
are admitted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

PERMISSIVE_LICENSES = frozenset(
    {"MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "ISC", "0BSD", "PSF-2.0"}
)


@dataclass(frozen=True)
class TaskSource:
    source_id: str
    hf_dataset: str
    revision: str
    dataset_license: str
    shards: tuple[str, ...]
    execution: str  # "native_venv" (runs on macOS without Docker) or "docker_required"
    notes: str = ""
    citation: str = ""
    columns: tuple[str, ...] = field(default_factory=tuple)

    def shard_urls(self) -> list[str]:
        base = f"https://huggingface.co/datasets/{self.hf_dataset}/resolve/{self.revision}/"
        return [base + shard for shard in self.shards]


SWE_SMITH_PY = TaskSource(
    source_id="swe-smith-py",
    hf_dataset="SWE-bench/SWE-smith-py",
    revision="77cab9055d42ab4a5c25c89a8f937096db13558e",
    dataset_license="MIT",
    shards=tuple(f"data/train-{i:05d}-of-00011.parquet" for i in range(11)),
    execution="native_venv",
    notes=(
        "Synthetic bugs injected into pinned public repositories. The `patch` "
        "column introduces the bug; its reverse is the reference fix. Task "
        "repositories are mirrored at github.com/swesmith/<owner>__<repo>.<sha8>."
    ),
    citation="arXiv:2504.21798",
    columns=("instance_id", "patch", "FAIL_TO_PASS", "PASS_TO_PASS", "repo", "problem_statement"),
)

SWE_REBENCH_V2 = TaskSource(
    source_id="swe-rebench-v2",
    hf_dataset="nebius/SWE-rebench-V2",
    revision="10483de0f50fe5da545942705a76c6150171af7f",
    dataset_license="CC-BY-4.0",
    shards=("data/train-00000-of-00001.parquet",),
    execution="docker_required",
    notes=(
        "Real GitHub issue/PR tasks with per-instance license fields. Grading "
        "requires the published x86 Docker images, so v1 records it as a "
        "license/dedup source only; it is not executed on the Mac route."
    ),
    citation="arXiv:2602.23866",
    columns=("instance_id", "repo", "license", "language", "created_at"),
)

SOURCES = {s.source_id: s for s in (SWE_SMITH_PY, SWE_REBENCH_V2)}

# Upstream licenses for SWE-smith repositories admitted by the v1 repository
# allowlist. Values came from the GitHub license API (collected 2026-10-31 in
# the companion license survey) and are re-checkable with
# ``scripts/learning_pipeline_v1.py check-licenses``.
REPO_LICENSES: dict[str, str] = {
    "Suor/funcy": "BSD-3-Clause",
    "alecthomas/voluptuous": "BSD-3-Clause",
    "arrow-py/arrow": "Apache-2.0",
    "hukkin/tomli": "MIT",
    "pytest-dev/iniconfig": "MIT",
    "life4/textdistance": "MIT",
    "mewwts/addict": "MIT",
    "dbader/schedule": "MIT",
    "un33k/python-slugify": "MIT",
    "luozhouyang/python-string-similarity": "MIT",
    "vi3k6i5/flashtext": "MIT",
    "martinblech/xmltodict": "MIT",
    "andialbrecht/sqlparse": "BSD-3-Clause",
    "msiemens/tinydb": "MIT",
    "python-hyper/h11": "MIT",
    "lepture/mistune": "BSD-3-Clause",
    "cknd/stackprinter": "MIT",
    "cool-RR/PySnooper": "MIT",
    "madzak/python-json-logger": "BSD-2-Clause",
    "erikrose/parsimonious": "MIT",
    "rustedpy/result": "MIT",
    "gweis/isodate": "BSD-3-Clause",
    "termcolor/termcolor": "MIT",
    "gruns/icecream": "MIT",
    "theskumar/python-dotenv": "BSD-3-Clause",
    "jd/tenacity": "Apache-2.0",
    "spulec/freezegun": "Apache-2.0",
    "google/textfsm": "Apache-2.0",
    "borntyping/python-colorlog": "MIT",
    "keleshev/schema": "MIT",
    "r1chardj0n3s/parse": "MIT",
    "chardet/chardet": "0BSD",
}


def upstream_repo(swesmith_repo: str) -> tuple[str, str]:
    """``swesmith/Suor__funcy.207a7810`` -> (``Suor/funcy``, ``207a7810``)."""
    name = swesmith_repo.split("/", 1)[1]
    base, _, sha8 = name.rpartition(".")
    owner, _, repo = base.partition("__")
    if not owner or not repo or not sha8:
        raise ValueError(f"unrecognized SWE-smith repo name: {swesmith_repo!r}")
    return f"{owner}/{repo}", sha8


def is_permissive(license_id: str | None) -> bool:
    return license_id in PERMISSIVE_LICENSES
