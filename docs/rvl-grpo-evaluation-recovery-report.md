# RVL GRPO evaluation failure recovery v1

## Question and decision

When candidate evaluation raises after loading a candidate into RVL's shared mutable trainer, does the adapter restore the active incumbent before releasing its model lock?

**Fixed and regression-covered with a fake CPU trainer/backend.** Evaluation now attempts incumbent restoration after exceptions and cancellation. The adapter invalidates its loaded-policy marker before calling the trainer restore method, so recovery forces a fresh incumbent load if candidate restore partially mutates state and then raises. If restoration itself fails, the adapter raises an explicit unknown-trainer-state error.

## Defect

The adapter previously restored the active policy only after every evaluation task completed. If backend generation or score computation raised, the active policy ID remained the incumbent while the shared trainer remained loaded with the candidate weights. A second edge case was a trainer restore that partially changed model/optimizer state and then raised: the cached loaded-policy marker could still equal the incumbent, causing recovery to incorrectly skip the actual restore.

## Verification and scope

`tests/test_rvl_grpo_hooks.py` injects both a generation failure after candidate weights are loaded and a candidate restore that mutates state before raising. It checks that the original error propagates after a forced incumbent restore, active identity remains the incumbent, and a later candidate evaluation still works. The focused group, freshness, concurrency, and RVL hook suite passed (17 tests), and the full suite passed (159 passed, 12 skipped) on Python 3.12.12.

This regression uses a fake trainer and backend. It does not validate cancellation scheduling, a failed restoration operation, another RVL revision, real model behavior, CUDA, or downstream task improvement. Existing real-trainer rollback smokes cover separate training-failure boundaries.
