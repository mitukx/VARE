"""Subprocess entry point: ``python -m vare.learning_pipeline.lora_worker CFG METRICS``.

Runs mlx-lm's LoRA trainer (``mlx_lm.lora.train_model``) with the same defaults as ``mlx_lm lora -c CFG``
and appends every train/valid report (loss, tokens/s, peak memory) to
METRICS as JSON lines. A separate process guarantees training memory is
released before evaluation starts.
"""
from __future__ import annotations

import json
import sys
import time
import types


def main(cfg_path: str, metrics_path: str) -> None:
    from mlx_lm import lora
    from mlx_lm.tuner.callbacks import TrainingCallback

    cfg = json.loads(open(cfg_path).read())
    args = dict(cfg)
    for k, v in lora.CONFIG_DEFAULTS.items():
        args.setdefault(k, v)
    args.setdefault("config", None)

    class Metrics(TrainingCallback):
        def _write(self, kind, info):
            with open(metrics_path, "a") as fh:
                fh.write(json.dumps({"kind": kind, "time": time.time(), **info}) + "\n")

        def on_train_loss_report(self, train_info):
            self._write("train", train_info)

        def on_val_loss_report(self, val_info):
            self._write("valid", val_info)

    # mlx_lm.lora.run() (0.32.2) replaces any passed training_callback with
    # its report_to callbacks, so the same steps are replicated here.
    import numpy as np

    ns = types.SimpleNamespace(**args)
    np.random.seed(ns.seed)
    model, tokenizer = lora.load(ns.model, tokenizer_config={"trust_remote_code": False})
    train_set, valid_set, _ = lora.load_dataset(ns, tokenizer)
    lora.train_model(ns, model, train_set, valid_set, Metrics())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
