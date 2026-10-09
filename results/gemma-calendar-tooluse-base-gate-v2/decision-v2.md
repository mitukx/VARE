# Gemma calendar base-gate result

## Decision

**STOP this model/task pairing.** The preregistered primary threshold required at least 4/8 exact successes in each of four operation families. The first complete family, `book_free_slot`, scored 0/8, so passing the primary threshold became impossible. No prompt, parser, task, budget, or threshold changes were made after evaluation began.

## Evidence

- Model: `google/gemma-2-2b-it`, pinned revision `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`; all nine frozen artifact hashes matched before load.
- Device/runtime: CPU, 8 threads, BF16, 2,614,341,888 parameters, 256,000-token tokenizer. Peak RSS was 5,992,448,000 bytes.
- Completed: 11/32 tasks (8 `book_free_slot`, 3 `move_conflict_then_book`); exact success 0/11.
- All 55 completed model outputs were wrapped in Markdown code fences. The frozen parser required raw JSON and rejected 55/55 outputs. Tool calls were consequently not executed. This is evidence of failure on the frozen response contract; it is not evidence about the model's underlying planning when given a permissive parser.
- Persisted completed-run time: 692.699 seconds, of which 687.5984 seconds was generation.
- One additional task (`cal-012`) was interrupted during generation and has no complete retained trace. No paid API, GPU, or external spend was used.

## Protocol deviation and limits

The v2 protocol required all 32 tasks to be recorded and did not predeclare an early futility stop. Execution stopped after 0/8 in the first family made the conjunctive primary threshold mathematically unreachable. The resulting resource gate is unmet; this is an explicitly incomplete run with a decisive primary failure, not a fully completed protocol. The interrupted task has no complete trace. Future protocols should predeclare a task-boundary futility rule and checkpoint each generated response if partial-task retention is needed.

This was a real pretrained model tested in a synthetic, procedurally generated task environment. It did not test any model update, RL effect, preference optimization, independent downstream capability gain, human-authored task performance, or general tool use. It contains no novelty claim. No retrial of this exact model/task cohort is warranted.

## Reproduction and artifacts

- Frozen protocol: `protocols/gemma_calendar_tooluse_base_gate_v2.lock.json`
- Runner and calibration: `scripts/run_gemma_calendar_tooluse_base_gate_v2.py`, `scripts/calibrate_gemma_calendar_tooluse_harness_v2.py`
- Raw partial task traces and exact states: `results/gemma-calendar-tooluse-base-gate-v2/run-1/result.json`
- Command, stdout/stderr, and execution metadata: `results/gemma-calendar-tooluse-base-gate-v2/execution-v2/`
- Raw result SHA-256: `4cb953ff951f906435154d9361797b83df3709e6c0ef432fb5dd5b795dbca4f7`
- Harness calibration passed before model evaluation: 32/32 known plans accepted; 32/32 initial unsolved states rejected; 32/32 protected-event operations rejected; 32/32 extra final states rejected; 8/8 cancel/recreate identity substitutions rejected; pinned single-user chat template rendered successfully.
