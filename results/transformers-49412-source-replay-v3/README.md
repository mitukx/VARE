# Transformers #49412 source replay v3

This version supersedes v2 after its exact-head review found that the v2 baseline classifier could accept an unrelated first-call error as a reproduced failure. V2's protocol, runner, output, and hashes remain unchanged and are explicitly not accepted as authoritative baseline reproduction.

The v3 baseline assertion first requires registration to return normally, then accepts only the exact expected conversion-mapping ValueError from the second registration. A separate fixed-source control injects an unrelated ValueError on the first call and confirms it is not interpreted as the baseline conflict. The fixed upstream source passed three properties and all three predeclared incorrect source mutations were detected.

This is narrow source-level evidence only. The probe imports the exact upstream fusion-mapping source file and replaces model discovery and registry persistence with fixed in-memory doubles. It does not execute candidate code, a hidden grader, the full upstream pytest suite, an agent, a model, or an optimizer. The task remains in the original frozen v1 heldout source allocation, is not reassigned, and is not eligible for model or agent use.

To reproduce, use an immutable Transformers clone and export the exact source module revisions:

```sh
mkdir -p /tmp/vare-49412-v3/baseline /tmp/vare-49412-v3/fixed
git -C /path/to/transformers show cb33194ad6152bd9fad6305378d92db385dd7b32:src/transformers/fusion_mapping.py > /tmp/vare-49412-v3/baseline/fusion_mapping.py
git -C /path/to/transformers show 01269bc99768ad45f8aeaeb701e1315371e25239:src/transformers/fusion_mapping.py > /tmp/vare-49412-v3/fixed/fusion_mapping.py
shasum -a 256 /tmp/vare-49412-v3/baseline/fusion_mapping.py /tmp/vare-49412-v3/fixed/fusion_mapping.py
```

The hashes must match `protocols/transformers_49412_source_replay_v3.json`. Run the six cases sequentially, once each, in fresh network-disabled containers with one CPU, 512 MiB memory, 64 PIDs, and a 30-second host timeout. The exact argument vectors and raw outputs are retained in `raw/<case>.json` and the adjacent stdout/stderr files. Each command must use the source SHA and expected classification in the protocol. For mutation cases the probe derives the exact frozen mutant in memory and reports the executed module SHA. The first-call-error control uses the fixed source SHA and injects only the specified test-double fault.

`freeze.json` records source, license, protocol, runner, mutation, and runtime image identities. Verify the bundle with `shasum -a 256 -c results/transformers-49412-source-replay-v3/SHA256SUMS` from the repository root. These commands replay the upstream registration function; they do not certify a repository-patch grader or model capability.
