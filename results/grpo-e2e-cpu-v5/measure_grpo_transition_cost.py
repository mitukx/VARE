"""Replay captured real-model scores; normalization microcost, not E2E speed."""
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import torch
from transformers.generation.utils import GenerationMixin

torch.set_num_threads(1)
data = torch.load(sys.argv[1], weights_only=True)
scores, tokens = data["scores"], data["tokens"]
proxy = SimpleNamespace(config=SimpleNamespace(get_text_config=lambda: SimpleNamespace(vocab_size=scores[0].shape[-1])))

def legacy():
    return GenerationMixin.compute_transition_scores(proxy, tokens, scores, normalize_logits=True)

def contiguous():
    return torch.stack([torch.log_softmax(s.float(), dim=-1).gather(1, tokens[:,i:i+1]).squeeze(1)
                        for i,s in enumerate(scores)], dim=1)

reference = torch.stack([torch.log_softmax(s.double(), dim=-1).gather(1, tokens[:,i:i+1]).squeeze(1)
                         for i,s in enumerate(scores)], dim=1)
for _ in range(20):
    legacy()
    contiguous()
times = {"legacy":[], "contiguous":[]}
for repeat in range(3):
    order = [("legacy",legacy),("contiguous",contiguous)]
    if repeat%2:
        order.reverse()
    for name, function in order:
        started = time.perf_counter()
        for _ in range(100):
            function()
        times[name].append((time.perf_counter()-started)/100)
result = {"scope":"Same captured neural-model score tensors; CPU one thread, 20 warmups, 3 alternating paired batches of 100 calls; no model generation or optimizer timing",
          "score_shapes":[list(s.shape) for s in scores], "seconds_per_call":times,
          "mean_s":{k:statistics.mean(v) for k,v in times.items()},
          "sample_sd_s":{k:statistics.stdev(v) for k,v in times.items()},
          "max_error_to_float64":{"legacy":float((legacy()-reference).abs().max()),
                                  "contiguous":float((contiguous()-reference).abs().max())}}
destination = Path(sys.argv[2])
with destination.open("x") as stream:
    stream.write(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
