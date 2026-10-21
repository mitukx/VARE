from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scripts.grade_transformers_fresh_dev_v1 import main
TASK_ID = 'transformers-49481-bloom-static-cache-v1'
if __name__ == '__main__':
    sys.argv[1:1] = ['--task-id', TASK_ID]
    raise SystemExit(main())
