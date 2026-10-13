# Artifact retention

Tracked files at or above 1 MiB (1,048,576 bytes) are frozen by [`protocols/tracked_large_blob_allowlist_v1.txt`](../protocols/tracked_large_blob_allowlist_v1.txt). The census has 235 blobs, about 1.53 GB, mostly adapter checkpoints and arrays under `results/`. They stay in git because they are retained raw records. This repository does not rewrite published history to remove them.

`scripts/check_tracked_blob_budget.py` fails if the current index adds, replaces, or drops one of those blobs. The script pins the allowlist's SHA-256. Extending the census to admit a new large checkpoint is a policy change, not the default path.

Git LFS and DVC are not used. Migrating objects that are already in published commits would rewrite history. New files at or above 1 MiB stay outside the git tree. Record their SHA-256 in the study manifest's `files` map, the same way existing manifests name retained outputs, and keep the manifest itself under the size threshold.

Check the tree with:

```bash
PYTHONPATH=src python3 scripts/check_tracked_blob_budget.py
```
