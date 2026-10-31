# Agent context-compaction retrospective replay v1

This artifact replays the two consumed fresh-Development agent transcripts through the deterministic context compactor. It reconstructs each original prompt and verifies that every recorded pre-generation token count matches under the pinned Qwen3-4B tokenizer before measuring the compacted final prompt.

| Attempt | Prompt before compaction | Prompt after compaction | Old complete rounds removed | Latest tool round kept |
| --- | ---: | ---: | ---: | --- |
| Transformers #49418 | 4,296 | 2,765 | 3 | yes |
| Ray #66694 | 4,140 | 2,795 | 2 | yes |

The tokenizer snapshot is `mlx-community/Qwen3-4B-Instruct-2507-4bit`, revision `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`. The original attempt records are hash-bound in the `attempt_sha256` fields of the task replay JSON files.

This is a deterministic prompt-history replay only. It ran zero model generations and zero grader calls, did not modify the frozen attempts, and does not establish that either model would have made a repair after compaction. No task exposure was added.

See `research/flagship/REPRODUCE.md` for the exact command and checksum verification.
