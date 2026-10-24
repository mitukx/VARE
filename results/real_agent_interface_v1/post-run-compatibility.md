# Post-run source compatibility notice

PR #125's frozen model execution used source commit
`d4752bff910ad64737b28b97d1000be81fa2aae8`. At that commit,
`src/vare/container_isolation.py` had SHA-256
`50e51865b094ab4a1666cfc961e9c435f9626b1952904dedb2ca85770b73837f`,
which accepts a raw local Docker image configuration ID. The v1 runner and E0
protocol also pin that exact implementation hash.

The CI repair restores `src/vare/container_isolation.py` byte-for-byte to the
PR #124 base version, SHA-256
`f6399f29f37d3d2c03c63f04c91efb25e60c31919ec3b83a8dda49cde6792e6a`,
so historical protocol graders continue to recognize their frozen source.
Consequently, replaying the v1 local-image experiment from the current branch
fails its intentional source-hash check and its raw local image identifier is
not accepted by the restored shared API. Reproduction of the already completed
run requires checking out the original pinned commit above and using its exact
protocol, image, model snapshot, and task inputs.

No frozen protocol, original run report, trajectory, calibration, or model
hash has been rewritten. Any future execution must use a new protocol and
source identities; it must not resume or reinterpret the consumed v1 run.
