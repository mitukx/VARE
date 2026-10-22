# accelerate-4296-skip-first-batches-v1

When training resumes from a skipped batch prefix, both sharded and dispatched loaders must preserve their configured iteration semantics and loader options. The current result silently changes configuration. Make the smallest compatible correction and preserve the remaining loader behavior.

Editable file: `src/accelerate/data_loader.py`.

This candidate task is not frozen for model inference. See `task.json` for source, license, execution and limitation provenance.
