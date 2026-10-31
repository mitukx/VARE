# Resource measurement erratum 001

This append-only correction applies to the vLLM #64830 source-qualification bundle. The original protocol, three attempt receipts, qualification report, and root `SHA256SUMS` are preserved unchanged. The JSON record beside this file binds the correction to the original bundle and receipt hashes.

The original `peak_process_tree_rss_bytes` value of 3,932,160 bytes in each attempt is the largest value observed by a 0.5-second `ps` polling loop. Each arm completed in about 0.53 seconds, so the recorded values do not establish OS-reported high-water memory or a complete process-tree peak. The earlier statement that all memory gates passed is withdrawn. The 4 GiB resource result is **INCONCLUSIVE**; a transient limit breach was not excluded.

The behavioral outcomes remain intact: baseline failed both intended assertions, the wrong mutant was rejected by both, and upstream-fixed passed both. This correction does not change those logs or classifications. It limits only the resource evidence.

The local reproduction started a child that waited 0.15 seconds, touched a 91 MiB allocation, and exited 0.05 seconds later. The first process-tree sample was 2,867,200 bytes; after a 0.5-second gap, `wait4` reported a direct-child high-water mark of 106,774,528 bytes. The corrected runner reports sampled process-tree RSS and this OS-reported direct-child high-water mark in separate fields. `wait4` is not a complete concurrent process-tree high-water mark, so future reports must state that limitation. The regression test uses a short-lived child allocation and checks that `wait4` retains its high-water after exit.

Do not rerun the consumed arms or reuse protocol v1. A future qualification or Development attempt requires a new frozen protocol bound to the corrected runner and an explicit resource-measurement scope.
