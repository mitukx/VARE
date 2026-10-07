from pathlib import Path
import sys

root = Path(sys.argv[1])
(root / "subject.py").write_text('''import math\n\ndef logsumexp(xs):\n    if not xs:\n        return -math.inf\n    m = max(xs)\n    return m + math.log(sum(math.exp(x - m) for x in xs))\n''', encoding='utf-8')
