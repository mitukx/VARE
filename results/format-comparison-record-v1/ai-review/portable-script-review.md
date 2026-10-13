# Portable witness: independent review notes

Inspected the in-progress `scripts/reproduce_format_comparison_record_v1.py`
and its tests on branch `codex/format-comparison-record-audit-20261013`.
The full portable replay was not rerun. Prior local record inspection remains
in `review.md` and `audit.json`.

## Concrete contract mismatch observed

At inspection time `parsed()` used `r"#### (-?\d+)"`, requiring exactly one
space. The frozen grader uses `r"^####\s+(-?\d+)\s*$"` on the last nonempty,
stripped line. The new test also explicitly rejected `"####  2"`.

Executed constructed counterexamples:

| text | portable parser | frozen parser |
| --- | --- | --- |
| `####  42` | None | 42 |
| `####\t42` | None | 42 |

The existing record's 18,400 eval and training output strings had zero parser
differences under these two rules. Thus the mismatch did not affect the
reported existing record result. It does affect the helper/test's stated
strict-format contract and should be corrected before treating it as a
reusable independent oracle.

The portable loop compared `strict_correct` only. A format-valid but incorrect
answer could have a false format judgment while retaining `strict_correct =
False` in both implementations. Comparing and reporting `strict_format`
independently would close this specific verification gap.

## Other static observations

- Retained progress is checked against the immutable decompressed SHA before
  analysis, and the archived artifact bytes are checked against the manifest.
- Output is labelled retrospective through the v3 summary call; operator and
  original source verification are explicitly false.
- All 80 format optimizer witnesses are checked for zero m/v and step counter.
  All 16 adapter hashes per seed must match the retained final adapter.
- The witness retains format-arm optimizer states and adapters. The earlier
  local checks of correctness/random checkpoint changes and equality remain
  local evidence unless those states or an independently usable reduced
  witness are also retained.
- This review does not authenticate the asserted chronological order of the
  frozen analysis commit and first result inspection.

No additional actionable issue was identified in the static inspection.

## Correction verification

Verified revision `148a429673c52da08834b9c6a9b9b60ea1913299` after the
earlier observations above. The parser now accepts the frozen whitespace
grammar. A bounded executable probe tested multiple spaces, tab separators,
trailing blank lines, incorrect but format-valid integers, and invalid
outputs. All six cases matched the frozen parser, strict-format decision,
and strict-correct decision. The original run was not reread for this check.

Static inspection confirms that the portable loop now compares both
`strict_correct` and `strict_format` and retains the format-valid output
count. The identified verification gap is closed. No further actionable
finding remains from this review.

Corrected executable artifact: `portable_parser_corrected.py` and
`portable_parser_corrected.json`.

Reviewed portable script SHA-256:
`458c89545368a53adc037b68d2d0d0322f6f569c93ec4d18028cc7b4c33f79e0`
