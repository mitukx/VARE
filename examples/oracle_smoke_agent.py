"""Harness-only external command agent used to test env-campaign plumbing."""
from pathlib import Path
import sys

workspace = Path(sys.argv[1])
(workspace / "subject.py").write_text('''import math\n\ndef logsumexp(xs):\n    if not xs:\n        return -math.inf\n    m = max(xs)\n    return m + math.log(sum(math.exp(x - m) for x in xs))\n''', encoding='utf-8')
print("patched smoke workspace")
