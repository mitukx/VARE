# Evidence packaging addendum

The matched-run outputs include two candidate workspaces whose `.git` directories preserve the baseline revision. Git cannot store a nested `.git` directory as ordinary repository files; the initial evidence commit therefore recorded those paths as gitlinks. This addendum preserves their exact bytes in `run-1/greedy/workspace-git-metadata.tar` and `run-1/sampled/workspace-git-metadata.tar`.

`workspace-git-archive-map.json` maps each `.git` member back to its original path in `run-1/evidence-manifest.json` and records the archive SHA-256. Extract an archive into its named workspace to restore those bytes. The original experiment manifest, report, protocol, candidate files, and model outputs are unchanged. `bundle-manifest-v2.json` covers the corrected tracked representation.
