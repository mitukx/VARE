# Evidence packaging addendum

The completed-run manifest includes two candidate workspaces with nested `.git` directories. The tracked evidence package stores those exact Git metadata files in `run-1/greedy/workspace-git-metadata.tar` and `run-1/sampled/workspace-git-metadata.tar`. The archives contain `.git/...` members; extract each archive into its corresponding `run-1/<arm>/workspace/` directory to restore them.

`workspace-git-archive-map.json` maps every archived member to the unchanged path and SHA-256 in `run-1/evidence-manifest.json`. That original run manifest and all model outputs are preserved. `bundle-manifest.json` covers every ordinary tracked evidence file and both Git-metadata archives.
