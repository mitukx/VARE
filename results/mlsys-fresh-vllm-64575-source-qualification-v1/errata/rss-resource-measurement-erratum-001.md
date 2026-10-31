# RSS resource measurement erratum 001

The original 30-file qualification bundle and its receipts remain unchanged. This append-only correction is bound to their hashes in the JSON record alongside this file.

The five stored `peak_process_tree_rss_bytes` values of 3,915,776 bytes are the largest values observed by the runner's 0.5-second `ps` sampling loop. Every attempt completed in about 0.53 seconds. A local reproduction sampled 4,046,848 bytes before a child touched 91 MiB; `wait4` reported a 107,511,808-byte direct-child high-water mark after a 0.5-second gap. The original runner therefore did not establish an OS high-water or complete process-tree peak. The earlier 4 GiB memory-pass interpretation is withdrawn; the resource result is **INCONCLUSIVE**.

The source behavior outcomes remain unchanged: baseline attempt-002 failed the two intended assertions and passed three controls; wrong-mutant v2 was rejected at its intended assertion and passed four controls; upstream-fixed passed all five cases. The first baseline setup failure and first wrong-mutant design failure stay preserved.

The corrected runner records sampled process-tree RSS separately from the OS `wait4` direct-child high-water. It explicitly does not claim a complete concurrent process-tree high-water mark. Do not rerun the consumed arms or reuse protocol v1; future execution requires a new frozen protocol and runner hash.
