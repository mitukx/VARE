import sys, json, argparse
sys.path.insert(0, "scripts")
import grpo_vertical_slice as gvs
from vare.learning_pipeline import grpo_mlx
from vare.learning_pipeline.prompting import build_messages
import mlx.core as mx, mlx.nn as nn

a = gvs.parse(["--run-dir", sys.argv[1], "--cache", sys.argv[2], "--tasks", "2"])
S = gvs.Slice(a)
tasks = S.train_tasks()
model, tok = S.load()
T = 0.7
res = []
for t in tasks:
    g = gvs.generate(model, tok, build_messages(t, t["_buggy"]), seed=1, temperature=T, top_p=0.95, max_tokens=256)
    s = grpo_mlx.Sample(instance_id=t["instance_id"], behavior_checkpoint_id="x", behavior_step=0,
                        prompt_ids=g["prompt_ids"], completion_ids=g["completion_ids"], reward=0.0,
                        verifier="v", sample_index=0)
    tokens, mask, start = grpo_mlx.sample_tensors(s, 0)
    n = len(s.completion_ids)
    def plain():
        lp = grpo_mlx.token_logprobs(model, tokens, temperature=T, start=start); mx.eval(lp); return lp[0, :n]
    def vg():
        def f(tokens):
            lp = grpo_mlx.token_logprobs(model, tokens, temperature=T, start=start)
            return (lp[0, :n] * 0).sum(), lp
        (l, lp), gr = nn.value_and_grad(model, f)(tokens); mx.eval(lp, gr); return lp[0, :n]
    import time
    def traced():
        def f(d):
            lp = grpo_mlx.token_logprobs(model, tokens, temperature=T, start=start) * d
            return lp.sum(), lp
        (_, lp), _g = mx.value_and_grad(f)(mx.array(1.0)); mx.eval(lp, _g); return lp[0, :n]
    t0=time.time(); A = plain(); ta=time.time()-t0; A2 = plain(); C = vg()
    t0=time.time(); E = traced(); te=time.time()-t0
    t0=time.time(); C = vg(); tc=time.time()-t0
    with gvs.grad_checkpointing(model.layers[0]):
        B = plain(); D = vg(); D2 = vg()
    def cmp(X):
        d = X - A
        return {"max_abs": float(mx.abs(d).max()), "mean_abs": float(mx.abs(d).mean()),
                "k3": float((mx.exp(-d) + d - 1).mean())}
    def cmp2(X, Y):
        d = X - Y
        return {"max_abs": float(mx.abs(d).max()), "k3": float((mx.exp(-d) + d - 1).mean())}
    r = {"n": n, "A2": cmp(A2), "B_ckpt_plain": cmp(B), "C_vg": cmp(C), "D_vg_ckpt": cmp(D),
         "D2_vs_D": cmp2(D2, D), "D_vs_C": cmp2(D, C), "E_vs_C": cmp2(E, C), "E_vs_D": cmp2(E, D), "sec": [ta, te, tc], "dtype": str(model.model.embed_tokens.scales.dtype) if hasattr(model.model.embed_tokens, "scales") else None}
    print(json.dumps(r), flush=True); res.append(r)
json.dump(res, open(sys.argv[1] + "/diag.json", "w"), indent=1)
