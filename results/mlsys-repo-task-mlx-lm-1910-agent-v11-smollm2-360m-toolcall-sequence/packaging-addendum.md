# Evidence packaging addendum

The run manifest includes two candidate workspaces with nested `.git` directories. Their exact metadata bytes are stored in `run-1/greedy/workspace-git-metadata.tar` and `run-1/sampled/workspace-git-metadata.tar`. Each archive contains `.git/...` members; extract it into the corresponding `run-1/<arm>/workspace/` directory to restore them.

`workspace-git-archive-map.json` maps every archived member to its unchanged run-manifest path and SHA-256. The original model outputs and `run-1/evidence-manifest.json` are preserved. `bundle-manifest.json` covers the ordinary tracked evidence, including both Git-metadata archives and the pre-inference verifier failure record.
