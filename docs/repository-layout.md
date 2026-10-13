# Repository layout and maintenance

The installed control-plane implementation is `src/vare/`. The root `vare/` package is a small checkout bootstrap pointing there; it is not a second implementation. Keep that path for standalone checkout commands. `pyproject.toml` defines the installed package layout and Python >=3.11 requirement.

| Path | Role | Maintenance rule |
| --- | --- | --- |
| `src/vare/` | Control plane, evaluation runner, durable stores, RVL integration | Implement changes here; inspect matching tests/contracts |
| `tests/` | Current regressions and optional dependency checks | Keep frozen comparison tests hash-stable; use a separate file for later coverage |
| `benchmarks/` | Protected graders, historical tasks, CPU fixtures | Task/lock/evaluator changes require their own versioned protocol |
| `protocols/` | Study specifications and outcome-bearing locks | Never revise an observed rule in place |
| `results/`, `evidence/` | Retained raw runs and evidence artifacts | Preserve negative results and artifact hashes; do not relocate frozen files |
| `docs/` | Reports, current guidance, and historical decisions | Use the [documentation index](README.md); label dated status snapshots |
| `scripts/`, `reproducers/` | Reproduction, audit, diagnostic entry points | Use the [script index](../scripts/README.md); scripts may require optional dependencies/platforms |
| `contributions/` | Upstream patches and review artifacts | A retained patch is not proof of upstream merge |
| `configs/`, `examples/` | Configuration and integration examples | Runnable examples are not measured research results |
| `minigrpo/`, `selfimprove/` | Experimental training/self-improvement packages | CPU tests exist; no retained model gain is established |
| `notebooks/` | Optional accelerator workflows | A notebook does not establish a completed accelerator run |
| `experiments/`, `data/`, `outputs/` | Historical/experimental assets | Consult the owning report/protocol before interpreting or changing them |
| `requirements/` | Dependency locks for specific reproduction paths | Install only the dependencies required by the selected path |

This organization adds navigation rather than moving research assets. Paths are part of many protocol/manifest identities. Published Git history and the 235 large-blob allowlist are preserved. Future generated checkpoints >=1 MiB belong outside Git, with their hashes recorded in a manifest; see [artifact retention](artifact-retention.md).

Before adding infrastructure, identify a concrete failure or measurement gap, find the existing owner, and implement the smallest change at that boundary. Training/serving internals belong in the integrated trainer when practical. Before starting model experiments, apply the admission and untouched-confirmation rules in [AGENTS.md](../AGENTS.md).
