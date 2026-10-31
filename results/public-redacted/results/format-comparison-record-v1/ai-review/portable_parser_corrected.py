"""Bounded whitespace counterexamples; no original progress or model reads."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('<SESSION_ROOT>/work/VARE-eval-stats')
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.reproduce_format_comparison_record_v1 import parsed
from vare.answer_scoring import score_output
from vare.heldout_exact_integer import parse_final_integer

cases = []
for text in ['####  42', '####\t42', 'work\n#### 42\n \t\n',
             '#### 42\nother', '', '#### +42']:
    actual = parsed(text)
    expected = parse_final_integer(text)
    scores = score_output('1 + 1', text)
    assert actual == expected
    assert (actual is not None) == scores['strict_format']
    assert (actual == 2) == scores['strict_correct']
    cases.append({'text': text, 'portable': actual, 'frozen': expected,
                  'strict_format': scores['strict_format'],
                  'strict_correct': scores['strict_correct']})
result = {'revision': '148a429673c52da08834b9c6a9b9b60ea1913299',
          'source_sha256': hashlib.sha256((ROOT / 'scripts/reproduce_format_comparison_record_v1.py').read_bytes()).hexdigest(),
          'cases': cases, 'all_checks_passed': True,
          'original_progress_read': False, 'model_or_training_called': False,
          'static_check': 'Both strict_correct and strict_format compared per eval row; format-valid total retained in audit report.'}
Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
