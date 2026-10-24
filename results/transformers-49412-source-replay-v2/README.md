# Transformers #49412 source replay v2

This bundle tests the pinned upstream `_register_module_fusion` implementation for repeated fusion registration. The first qualification bundle's baseline pytest failure was in test setup: `setUp` tried to patch `_REGISTERED_FUSION_SOURCE_PATTERNS`, which does not exist at the baseline revision. The direct probe here bypasses that test fixture and exercises the target function with fixed in-memory model-discovery and mapping-registry doubles.

The replay found the expected baseline conflict, the fixed source passed all three behavioral properties, and all three predeclared incorrect mutations were detected. Every invocation used the frozen network-disabled container limits. The raw stdout, stderr, command arguments, exit status, wall time, and host swap snapshots are retained under `raw/`.

This is source-level behavior evidence only. It does not certify a candidate execution boundary, a repository-patch grader, task eligibility, a model trajectory, or policy improvement. The task remains in the original frozen v1 heldout source allocation and is not reassigned to Development or exposed to a model or agent.

To reproduce, use an immutable Transformers clone and create temporary source directories outside the candidate workspace:

```sh
mkdir -p /tmp/vare-49412-v2/baseline /tmp/vare-49412-v2/fixed
git -C /path/to/transformers show cb33194ad6152bd9fad6305378d92db385dd7b32:src/transformers/fusion_mapping.py > /tmp/vare-49412-v2/baseline/fusion_mapping.py
git -C /path/to/transformers show 01269bc99768ad45f8aeaeb701e1315371e25239:src/transformers/fusion_mapping.py > /tmp/vare-49412-v2/fixed/fusion_mapping.py
shasum -a 256 /tmp/vare-49412-v2/baseline/fusion_mapping.py /tmp/vare-49412-v2/fixed/fusion_mapping.py
```

The two hashes must match `protocols/transformers_49412_source_replay_v2.json`. Use the exact probe at `scripts/reproduce_transformers_49412_source_replay_v2.py`. Run the five cases once each in fresh containers with `--network none`, `--cpus 1`, `--memory 536870912`, `--pids-limit 64`, and a 30-second host timeout. The exact argument vectors used in this run are preserved in each `raw/<case>.json`; the fixed source is supplied for all three mutant cases, and the probe applies the frozen single-anchor mutation in memory. Each case must use the expected source SHA from the protocol and produce the expected classification and executed-module SHA recorded in its raw stdout.

The source files are not copied into this repository. `freeze.json` records upstream revisions, source/license/probe/protocol hashes, mutation hashes, the runtime image identity, and the resource envelope. Verify retained files with `shasum -a 256 -c results/transformers-49412-source-replay-v2/SHA256SUMS` from the repository root. These commands replay implementation behavior; they do not run the original upstream pytest test or an isolated candidate task.
