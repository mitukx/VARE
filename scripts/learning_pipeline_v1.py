#!/usr/bin/env python3
"""CLI for coding-policy learning v1 (Qwen3-4B 4bit LoRA on MLX, one Mac).

  build-curriculum  pinned SWE-smith-py rows -> train/held-out manifests
  check-licenses    print the dataset/repository license table
  freeze            write frozen input/code hashes into the protocol
  verify            recompute and check frozen hashes
  memory-estimate   analytic LoRA training memory estimate
  preflight         one-task disk/runtime/memory probe + stop rules
  calibrate | rollout | train | evaluate | compare | run-all

``--mode mock`` (MockPolicy, no training) and ``--mode synthetic_model``
(a tiny random MLX model passed with ``--model``) produce artifacts labeled
``"synthetic": true``. Only ``--mode real`` results may be reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vare.learning_pipeline import curriculum, exclusions, pipeline, resources, sources  # noqa: E402
from vare.learning_pipeline.resources import GiB, StopRule  # noqa: E402

CONFIG_DIR = ROOT / "configs" / "learning_pipeline_v1"
DEFAULT_PROTOCOL = ROOT / "protocols" / "coding_policy_learning_mlx_4b_v1.json"
DEFAULT_CACHE = Path.home() / ".cache" / "vare-learning-pipeline-v1"
DEFAULT_RUN = ROOT / "artifacts" / "learning_pipeline_v1" / "run-001"

FROZEN_CODE = sorted(
    str(p.relative_to(ROOT)) for p in (ROOT / "src" / "vare" / "learning_pipeline").glob("*.py")
) + ["src/vare/eval_stats.py", "scripts/learning_pipeline_v1.py"]
FROZEN_INPUTS = [
    "configs/learning_pipeline_v1/curriculum.json",
    "configs/learning_pipeline_v1/repos.lock.json",
    "configs/learning_pipeline_v1/train.manifest.jsonl",
    "configs/learning_pipeline_v1/heldout.manifest.jsonl",
    "requirements/mlsys-agent-mlx.lock.txt",
    "requirements/learning-pipeline-v1.txt",
]


def cmd_build(a) -> int:
    cfg = json.loads((CONFIG_DIR / "curriculum.json").read_text())
    lock = json.loads((CONFIG_DIR / "repos.lock.json").read_text())
    report = curriculum.build(cfg=cfg, repo_lock=lock, vare_root=ROOT, cache=Path(a.cache),
                              out_dir=CONFIG_DIR, materialize=not a.no_materialize)
    (CONFIG_DIR / "build_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("counts", "per_repository", "manifest_sha256")}, indent=2))
    return 0


def cmd_licenses(a) -> int:
    cfg = json.loads((CONFIG_DIR / "curriculum.json").read_text())
    bad = 0
    print(f"dataset {sources.SWE_SMITH_PY.hf_dataset}@{sources.SWE_SMITH_PY.revision[:12]}: "
          f"{sources.SWE_SMITH_PY.dataset_license}")
    for s in sources.SOURCES.values():
        if s.execution != "native_venv":
            print(f"license-only source {s.hf_dataset}: {s.dataset_license} ({s.execution})")
    scan = exclusions.scan_vare_tree(ROOT, skip=cfg.get("scan_skip", ()))
    excluded = exclusions.excluded_repositories(scan)
    for split in ("train_repositories", "heldout_repositories"):
        for r in cfg[split]:
            lic = sources.REPO_LICENSES.get(r)
            why = exclusions.is_excluded(r, excluded) or (None if sources.is_permissive(lic) else "license")
            bad += why is not None
            print(f"{split[:-13]:8s} {r:45s} {lic or 'UNKNOWN':14s} {'EXCLUDED: ' + why if why else 'ok'}")
    missing = exclusions.check_required(scan)
    print(f"exclusion scan: {scan['files_scanned']} files, {len(scan['identifiers'])} identifiers, "
          f"required missing={missing}")
    return 1 if bad or missing else 0


def cmd_freeze(a) -> int:
    p = Path(a.protocol)
    proto = json.loads(p.read_text())
    proto["frozen"] = {"inputs": pipeline.code_hashes(ROOT, FROZEN_INPUTS),
                       "code": pipeline.code_hashes(ROOT, FROZEN_CODE)}
    p.write_text(json.dumps(proto, indent=2) + "\n")
    print(f"froze {len(FROZEN_INPUTS)} inputs and {len(FROZEN_CODE)} code files into {p}")
    return 0


def cmd_verify(a) -> int:
    proto = json.loads(Path(a.protocol).read_text())
    print(json.dumps(pipeline.verify_protocol(proto, ROOT)))
    return 0


def cmd_memory(a) -> int:
    for seq in (2048, 3072, 4096):
        e = resources.estimate_lora_training_bytes(weights_bytes=int(2.26e9), seq_len=seq, vocab=151936,
                                                   hidden=2560, n_layers=36, lora_layers=a.lora_layers, rank=8)
        print(seq, {k: round(v / GiB, 2) for k, v in e.items()})
    print("now:", resources.memory_snapshot())
    return 0


def make_run(a) -> pipeline.Run:
    return pipeline.Run(Path(a.protocol), ROOT, Path(a.run_dir), Path(a.cache), mode=a.mode,
                        model_override=a.model)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN))
    ap.add_argument("--mode", choices=("real", "synthetic_model", "mock"), default="real")
    ap.add_argument("--model", default=None, help="local model path override (synthetic_model mode)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-curriculum")
    b.add_argument("--no-materialize", action="store_true")
    sub.add_parser("check-licenses")
    sub.add_parser("freeze")
    sub.add_parser("verify")
    m = sub.add_parser("memory-estimate")
    m.add_argument("--lora-layers", type=int, default=8)
    pf = sub.add_parser("preflight")
    pf.add_argument("--no-train-smoke", action="store_true")
    for name in ("calibrate", "rollout", "train", "evaluate", "compare"):
        sub.add_parser(name)
    ra = sub.add_parser("run-all")
    ra.add_argument("--skip-preflight", action="store_true")
    ra.add_argument("--no-train-smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.mode == "real" and a.model:
        ap.error("--model is only for synthetic_model mode; real mode uses the protocol's pinned model")
    simple = {"build-curriculum": cmd_build, "check-licenses": cmd_licenses, "freeze": cmd_freeze,
              "verify": cmd_verify, "memory-estimate": cmd_memory}
    try:
        if a.cmd in simple:
            return simple[a.cmd](a)
        run = make_run(a)
        if a.cmd == "preflight":
            pipeline.verify_protocol(run.protocol, ROOT)
            out = run.preflight(train_smoke=not a.no_train_smoke)
            print(json.dumps(out["projection"], indent=2))
        elif a.cmd == "run-all":
            out = run.run_all(skip_preflight=a.skip_preflight, train_smoke=not a.no_train_smoke)
            print(json.dumps({k: v["verdict"] for k, v in out["comparisons"].items()}, indent=2))
        else:
            out = getattr(run, a.cmd)()
            print(json.dumps(out, indent=2, default=str)[:4000])
    except StopRule as exc:
        print(str(exc), file=sys.stderr)
        stop_dir = Path(a.run_dir)
        stop_dir.mkdir(parents=True, exist_ok=True)
        (stop_dir / "STOPPED.json").write_text(json.dumps({"rule": exc.rule, "detail": exc.detail}, indent=2,
                                                          default=str) + "\n")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
