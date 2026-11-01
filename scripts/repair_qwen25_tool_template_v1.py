#!/usr/bin/env python3
"""Export a prospective template correction without changing cached snapshots.

The exact cached Qwen2.5 example contains doubled literal braces. This tool
only repairs that one known pattern; it never repairs model-generated JSON.
"""
import argparse
import hashlib
import json
from pathlib import Path

BAD_EXAMPLE = r'{{\"name\": <function-name>, \"arguments\": <args-json-object>}}'
GOOD_EXAMPLE = BAD_EXAMPLE[1:-1]


def repair(config: dict) -> dict:
    template = config.get("chat_template")
    if not isinstance(template, str) or template.count(BAD_EXAMPLE) != 1:
        raise ValueError("expected exactly one known doubled-brace example")
    corrected = dict(config)
    corrected["chat_template"] = template.replace(BAD_EXAMPLE, GOOD_EXAMPLE, 1)
    return corrected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    before = args.source.read_bytes()
    config = json.loads(before)
    corrected = repair(config)
    encoded = (json.dumps(corrected, indent=2, ensure_ascii=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(encoded)
    print(json.dumps({"source_sha256": hashlib.sha256(before).hexdigest(),
                      "output_sha256": hashlib.sha256(encoded).hexdigest(),
                      "changed_field": "chat_template", "model_inference": "NOT_RUN"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
