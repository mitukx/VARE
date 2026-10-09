# Independent audit

An independent review checked the frozen v4 protocol/script and pinned source hashes against both raw runs. All matched. Baseline and candidate result JSON were identical, both exited 0, and all four GA/source outputs reported `device: cpu`.

The reviewer verified the key trace:

- GA=1: row 10 is trained at version 0; row 11 was collated at version 0 and trained at version 1; two optimizer steps completed and the scalar weight moved `0.1000 → 0.0990 → 0.0985`.
- GA=2: rows 10 and 11 are both trained at version 0 in one optimizer update; the scalar weight moved `0.1000 → 0.0990`.

The source order and pinned Accelerate prefetch explain the check/use timing. The review agrees this is a conditional synthetic Trainer-loop finding, not proof of an unambiguous contract violation or real-policy improvement. It identified one minor protocol wording mismatch: the lock says the script requires CPU, while the script configures `use_cpu=True` and records the selected device without a literal runtime assertion. The recorded outputs all select CPU; the runner's runtime assertion remains absent.
