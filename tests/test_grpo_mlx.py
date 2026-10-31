"""Objective-level tests for the MLX GRPO update (synthetic toy models only).

The group/lag logic runs everywhere. Tests that need MLX run on a CPU MLX
install (CI job ``grpo-mlx-objective-cpu``) and are skipped otherwise; none of
them loads a pretrained model.
"""
from __future__ import annotations

import importlib.util
import math

import pytest

from vare.learning_pipeline import grpo_mlx as G

HAS_MLX = importlib.util.find_spec("mlx") is not None and importlib.util.find_spec("mlx_lm") is not None
needs_mlx = pytest.mark.skipif(not HAS_MLX, reason="mlx / mlx-lm not installed")


def S(iid, idx, comp, reward, *, ckpt="c0", step=0, prompt=(1, 2), verifier="v1"):
    return G.Sample(instance_id=iid, behavior_checkpoint_id=ckpt, behavior_step=step, prompt_ids=list(prompt),
                    completion_ids=list(comp), reward=float(reward), verifier=verifier, sample_index=idx)


# ------------------------------------------------------------ pure python
def test_group_advantages_normalized_and_centered():
    adv, inf = G.group_advantages([1, 0, 0, 0])
    std = math.sqrt(0.1875)
    assert inf
    assert adv == pytest.approx([0.75 / std, -0.25 / std, -0.25 / std, -0.25 / std])
    adv2, _ = G.group_advantages([1, 0, 0, 0], normalize_std=False)
    assert adv2 == pytest.approx([0.75, -0.25, -0.25, -0.25])
    assert sum(adv) == pytest.approx(0.0)


@pytest.mark.parametrize("rewards", [[0, 0, 0, 0], [1, 1, 1, 1], [0.5]])
def test_zero_variance_groups_have_zero_advantage(rewards):
    adv, inf = G.group_advantages(rewards)
    assert adv == [0.0] * len(rewards) and not inf


def test_group_validation():
    with pytest.raises(G.GroupError, match="duplicate"):
        G.group_samples([S("a", 0, [3], 1), S("a", 0, [4], 0)])
    with pytest.raises(G.GroupError, match="verifier"):
        G.group_samples([S("a", 0, [3], 1), S("a", 1, [4], 0, verifier="v2")])
    with pytest.raises(G.GroupError, match="partial"):
        G.group_samples([S("a", 0, [3], 1), S("a", 1, [4], 0)], group_size=4)
    with pytest.raises(G.GroupError, match="empty completion"):
        G.group_samples([S("a", 0, [], 1)])
    g = G.group_samples([S("a", 0, [3], 1), S("a", 0, [3], 1, ckpt="c1"), S("b", 0, [3], 0)])
    assert set(g) == {("a", "c0"), ("a", "c1"), ("b", "c0")}  # behavior policy is part of the group key


def test_policy_lag_rules():
    on = [S("a", 0, [3], 1, ckpt="c5", step=5)]
    assert G.check_policy_lag(on, policy_checkpoint_id="c5", policy_step=5, max_policy_lag=0) == 0
    with pytest.raises(G.StalePolicyError):
        G.check_policy_lag(on, policy_checkpoint_id="other", policy_step=5, max_policy_lag=0)
    with pytest.raises(G.StalePolicyError):
        G.check_policy_lag(on, policy_checkpoint_id="c6", policy_step=6, max_policy_lag=0)
    assert G.check_policy_lag(on, policy_checkpoint_id="c6", policy_step=6, max_policy_lag=1) == 1
    with pytest.raises(G.StalePolicyError, match="future"):
        G.check_policy_lag(on, policy_checkpoint_id="c4", policy_step=4, max_policy_lag=3)


# -------------------------------------------------------------- toy models
def _bigram(V=6, seed=0):
    import mlx.core as mx
    import mlx.nn as nn

    class Bigram(nn.Module):
        def __init__(self):
            super().__init__()
            self.table = nn.Embedding(V, V)

        def __call__(self, x):
            return self.table(x)

    mx.random.seed(seed)
    m = Bigram()
    mx.eval(m.parameters())
    return m


def _qwen3(tied=True, seed=0, vocab=40):
    import mlx.core as mx
    from mlx_lm.models import qwen3

    args = qwen3.ModelArgs(model_type="qwen3", hidden_size=32, num_hidden_layers=2, intermediate_size=48,
                           num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=vocab, num_key_value_heads=2,
                           max_position_embeddings=256, head_dim=8, tie_word_embeddings=tied, rope_theta=10000.0)
    mx.random.seed(seed)
    m = qwen3.Model(args)
    mx.eval(m.parameters())
    return m


def _np_logsoftmax(row):
    import numpy as np

    row = np.asarray(row, dtype=np.float64)
    z = row - row.max()
    return z - np.log(np.exp(z).sum())


@needs_mlx
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_token_logprobs_match_hand_computation(temperature):
    import numpy as np

    m = _bigram()
    W = np.asarray(m.table.weight, dtype=np.float64) / temperature
    seq = [0, 3, 1, 4, 4, 2]
    tokens, _ = G.build_batch([(seq[:2], seq[2:])], pad_id=5)
    lp = np.asarray(G.token_logprobs(m, tokens, temperature=temperature))[0]
    want = [_np_logsoftmax(W[seq[t]])[seq[t + 1]] for t in range(len(seq) - 1)]
    assert lp == pytest.approx(want, abs=1e-5)


@needs_mlx
def test_mask_excludes_prompt_and_padding():
    import numpy as np

    tokens, mask = G.build_batch([([7, 8, 9], [1, 2]), ([7], [3, 4, 5, 6])], pad_id=0)
    assert np.asarray(tokens).tolist() == [[7, 8, 9, 1, 2], [7, 3, 4, 5, 6]]
    # target t is tokens[:, t+1]; only completion targets count
    assert np.asarray(mask).tolist() == [[0, 0, 1, 1], [1, 1, 1, 1]]
    tokens, mask = G.build_batch([([7, 8], [1]), ([7], [3, 4, 5])], pad_id=0)
    assert np.asarray(tokens).tolist() == [[7, 8, 1, 0], [7, 3, 4, 5]]
    assert np.asarray(mask).tolist() == [[0, 1, 0], [1, 1, 1]]


@needs_mlx
@pytest.mark.parametrize("tied", [True, False])
def test_padding_and_prompt_slicing_do_not_change_completion_logprobs(tied):
    import numpy as np

    m = _qwen3(tied=tied)
    a = ([1, 2, 3, 4, 5], [6, 7])
    b = ([1, 9], [10, 11, 12, 13, 14, 15])
    tok, mask = G.build_batch([a, b], pad_id=0)
    full = np.asarray(G.token_logprobs(m, tok))
    for row, (p, c) in enumerate([a, b]):
        t1, m1 = G.build_batch([(p, c)], pad_id=0)
        alone = np.asarray(G.token_logprobs(m, t1))[0]
        sl = np.asarray(G.token_logprobs(m, t1, start=len(p) - 1))[0]
        keep = np.asarray(mask)[row] > 0
        assert full[row][keep] == pytest.approx(alone[len(p) - 1:len(p) - 1 + len(c)], abs=1e-4)
        assert sl == pytest.approx(alone[len(p) - 1:], abs=1e-4)


@needs_mlx
def test_clipped_surrogate_values_and_gradients():
    import mlx.core as mx

    blp = mx.zeros((1, 3))
    ratio = mx.array([[1.5, 0.5, 1.0]])
    mask = mx.ones((1, 3))
    for a, want in [(1.0, -(1.2 + 0.5 + 1.0) / 3), (-1.0, (1.5 + 0.8 + 1.0) / 3)]:
        def f(lp, a=a):
            return G.clipped_surrogate(lp, blp, mx.array([a]), mask, eps_low=0.2, eps_high=0.2, n_tokens=3.0)[0]

        lp = mx.log(ratio)
        assert float(f(lp)) == pytest.approx(want, abs=1e-6)
        g = mx.grad(f)(lp)[0].tolist()
        if a > 0:  # ratio 1.5 clipped above -> no gradient; 0.5 not clipped for A>0
            assert g == pytest.approx([0.0, -0.5 / 3, -1.0 / 3], abs=1e-6)
        else:  # ratio 0.5 clipped below for A<0 -> no gradient; 1.5 not clipped
            assert g == pytest.approx([1.5 / 3, 0.0, 1.0 / 3], abs=1e-6)
    # masked tokens contribute nothing
    m2 = mx.array([[1.0, 0.0, 1.0]])
    l2 = G.clipped_surrogate(mx.log(ratio), blp, mx.array([1.0]), m2, eps_low=0.2, eps_high=0.2, n_tokens=2.0)[0]
    assert float(l2) == pytest.approx(-(1.2 + 1.0) / 2, abs=1e-6)


def _samples_with_lp(model, samples, temperature=1.0):
    G.compute_behavior_logprobs(model, samples, temperature=temperature)
    return samples


@needs_mlx
def test_on_policy_gradient_matches_float64_reference():
    """On-policy (rho == 1) gradient == -(1/N) sum_i A_i sum_t grad log pi."""
    import numpy as np

    V = 6
    m = _bigram(V)
    W = np.asarray(m.table.weight, dtype=np.float64)
    samples = [S("a", 0, [3, 4], 1, prompt=[0, 1]), S("a", 1, [2], 0, prompt=[0, 1]),
               S("a", 2, [5, 5, 3], 0, prompt=[0, 1]), S("b", 0, [1], 1, prompt=[2]), S("b", 1, [4], 1, prompt=[2])]
    _samples_with_lp(m, samples)
    cfg = G.GRPOConfig()
    groups = G.group_samples(samples)
    work = []
    for key in sorted(groups):
        g = sorted(groups[key], key=lambda s: s.sample_index)
        adv, inf = G.group_advantages([s.reward for s in g])
        if inf:
            work += list(zip(g, adv))
    n = sum(len(s.completion_ids) for s, _ in work)
    assert n == 6  # group b is zero-variance: excluded from the token denominator
    grads, diag = G.policy_gradient(m, work, cfg, n_tokens=n)
    assert diag["ratio_mean"] == pytest.approx(1.0, abs=1e-6) and diag["clip_fraction"] == 0.0
    ref = np.zeros_like(W)
    for s, a in work:
        seq = s.prompt_ids + s.completion_ids
        for t in range(len(s.prompt_ids) - 1, len(seq) - 1):
            x, y = seq[t], seq[t + 1]
            p = np.exp(_np_logsoftmax(W[x]))
            onehot = np.eye(V)[y]
            ref[x] += -(a / n) * (onehot - p)
    assert np.asarray(grads["table"]["weight"]) == pytest.approx(ref, abs=1e-6)


@needs_mlx
def test_gradient_sign_raises_rewarded_completion():
    import mlx.core as mx
    import mlx.optimizers as optim

    m = _bigram()
    prompt = [0]

    def lp(tok):
        t, _ = G.build_batch([(prompt, [tok])], pad_id=0)
        return float(G.token_logprobs(m, t)[0, 0])

    good0, bad0 = lp(3), lp(4)
    samples = _samples_with_lp(m, [S("a", 0, [3], 1, prompt=prompt), S("a", 1, [4], 0, prompt=prompt)])
    opt = optim.SGD(learning_rate=0.5)
    rec = G.train_step(m, opt, samples, G.GRPOConfig(max_grad_norm=None), policy_checkpoint_id="c0",
                       policy_step=0)
    mx.eval(m.parameters())
    assert rec["update_applied"] and rec["params_changed"]
    assert lp(3) > good0 and lp(4) < bad0


@needs_mlx
def test_zero_variance_step_applies_no_update():
    import mlx.optimizers as optim

    m = _bigram()
    samples = _samples_with_lp(m, [S("a", 0, [3], 1), S("a", 1, [4], 1), S("b", 0, [3], 0), S("b", 1, [2], 0)])
    opt = optim.Adam(learning_rate=0.1)
    before = G.trainable_digest(m)
    rec = G.train_step(m, opt, samples, G.GRPOConfig(), policy_checkpoint_id="c0", policy_step=0)
    assert rec["update_applied"] is False and rec["reason"] == "no_informative_group"
    assert rec["n_informative_groups"] == 0 and G.trainable_digest(m) == before
    assert int(opt.state["step"].item()) == 0


@needs_mlx
def test_zero_variance_group_does_not_change_gradient():
    import numpy as np

    m = _bigram()
    mixed = [S("a", 0, [3], 1, prompt=[0]), S("a", 1, [4], 0, prompt=[0])]
    flat = [S("b", 0, [1, 2, 3], 1, prompt=[5]), S("b", 1, [2], 1, prompt=[5])]
    _samples_with_lp(m, mixed + flat)
    cfg = G.GRPOConfig(max_grad_norm=None)
    import mlx.optimizers as optim

    def step(samples):
        mm = _bigram()
        opt = optim.SGD(learning_rate=1.0)
        G.train_step(mm, opt, samples, cfg, policy_checkpoint_id="c0", policy_step=0)
        return np.asarray(mm.table.weight)

    assert step(mixed + flat) == pytest.approx(step(mixed), abs=1e-7)


@needs_mlx
def test_stale_policy_ratio_and_rejection():
    import mlx.optimizers as optim

    m = _bigram()
    samples = _samples_with_lp(m, [S("a", 0, [3, 1], 1, prompt=[0]), S("a", 1, [4, 2], 0, prompt=[0])])
    opt = optim.SGD(learning_rate=5.0)
    cfg = G.GRPOConfig(max_grad_norm=None)
    rec0 = G.train_step(m, opt, samples, cfg, policy_checkpoint_id="c0", policy_step=0)
    assert rec0["ratio_mean"] == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(G.StalePolicyError):  # on-policy default refuses reuse
        G.train_step(m, opt, samples, cfg, policy_checkpoint_id="c1", policy_step=1)
    lagged = G.GRPOConfig(max_grad_norm=None, max_policy_lag=1)
    rec1 = G.train_step(m, opt, samples, lagged, policy_checkpoint_id="c1", policy_step=1)
    assert rec1["max_observed_lag"] == 1
    assert rec1["ratio_max"] > 1.2 and rec1["clip_fraction"] > 0  # importance ratio is live and clipped


def test_informative_samples_selects_mixed_groups_only():
    out = G.informative_samples([S("a", 0, [3], 1), S("a", 1, [4], 0), S("b", 0, [3], 1), S("b", 1, [4], 1)],
                                G.GRPOConfig())
    assert [(s.instance_id, s.sample_index) for s in out] == [("a", 0), ("a", 1)]


@needs_mlx
def test_missing_behavior_logprobs_rejected():
    import mlx.optimizers as optim

    m = _bigram()
    with pytest.raises(ValueError, match="behavior log-probs"):
        G.train_step(m, optim.SGD(learning_rate=0.1), [S("a", 0, [3], 1), S("a", 1, [4], 0)], G.GRPOConfig(),
                     policy_checkpoint_id="c0", policy_step=0)


def _lora_qwen3(lora_seed=0):
    m = _qwen3(seed=0)  # fixed "pretrained" base weights
    G.make_lora_policy(m, num_layers=2, rank=4, scale=10.0, seed=lora_seed)
    return m


@needs_mlx
def test_on_policy_ratio_is_exactly_one_through_the_update_path():
    """Behavior log-probs use the traced loss the update differentiates, also
    under mlx-lm gradient checkpointing, so a lag-0 ratio is 1 to float
    precision and never clipped. Advantages do not change the log-probs."""
    import mlx.optimizers as optim
    from mlx_lm.tuner.trainer import grad_checkpoint

    m = _lora_qwen3(lora_seed=3)
    samples = [S("a", 0, [5, 9, 11, 2], 1, prompt=[1, 7, 3]), S("a", 1, [6, 6], 0, prompt=[1, 7, 3]),
               S("b", 0, [4, 8, 30], 0, prompt=[2]), S("b", 1, [13], 1, prompt=[2])]
    cfg = G.GRPOConfig(temperature=0.7, max_grad_norm=None)
    layer_cls = type(m.layers[0])
    original = layer_cls.__call__
    try:
        grad_checkpoint(m.layers[0])
        G.compute_behavior_logprobs(m, samples, cfg)
        first = [list(s.behavior_logprobs) for s in samples]
        G.compute_behavior_logprobs(m, samples, cfg)
        assert [s.behavior_logprobs for s in samples] == first  # deterministic
        rec = G.train_step(m, optim.SGD(learning_rate=0.1), samples, cfg, policy_checkpoint_id="c0", policy_step=0)
    finally:
        layer_cls.__call__ = original
    assert rec["update_applied"] and rec["params_changed"]
    assert rec["max_abs_log_ratio"] <= 1e-6
    assert rec["ratio_mean"] == pytest.approx(1.0, abs=1e-6) and rec["clip_fraction"] == 0.0
    assert rec["approx_kl"] == pytest.approx(0.0, abs=1e-9)
    # untraced forward agrees on CPU float32 (on Metal bf16 it may not, which is why it is not used)
    for s, lp in zip(samples, first):
        assert G.forward_logprobs(_lora_qwen3(lora_seed=3), s, temperature=0.7) == pytest.approx(lp, abs=1e-4)


@needs_mlx
def test_behavior_logprobs_require_cfg_or_temperature():
    with pytest.raises(ValueError):
        G.compute_behavior_logprobs(_bigram(), [S("a", 0, [3], 1)])


@needs_mlx
def test_lora_update_is_real_and_context_dependent():
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten

    m = _lora_qwen3()
    frozen0 = {k: np.asarray(v) for k, v in tree_flatten(m.parameters()) if "lora" not in k}
    trainable = [k for k, _ in tree_flatten(m.trainable_parameters())]
    assert trainable and all("lora_" in k for k in trainable)
    p1, p2 = [1, 2, 3], [4, 5, 6]

    def lp(prompt, tok):
        t, _ = G.build_batch([(prompt, [tok])], pad_id=0)
        return float(G.token_logprobs(m, t, start=len(prompt) - 1)[0, 0])

    base = (lp(p1, 7) - lp(p1, 8), lp(p2, 8) - lp(p2, 7))
    opt = optim.Adam(learning_rate=1e-2)
    for step in range(15):
        ck = f"c{step}"
        samples = [S("t1", 0, [7], 1, prompt=p1, ckpt=ck, step=step), S("t1", 1, [8], 0, prompt=p1, ckpt=ck, step=step),
                   S("t2", 0, [7], 0, prompt=p2, ckpt=ck, step=step), S("t2", 1, [8], 1, prompt=p2, ckpt=ck, step=step)]
        _samples_with_lp(m, samples)
        rec = G.train_step(m, opt, samples, G.GRPOConfig(), policy_checkpoint_id=ck, policy_step=step)
        assert rec["update_applied"] and rec["params_changed"]
    mx.eval(m.parameters())
    after = (lp(p1, 7) - lp(p1, 8), lp(p2, 8) - lp(p2, 7))
    # the same completion token is pushed up under one prompt and down under the other
    assert after[0] > base[0] + 0.5 and after[1] > base[1] + 0.5
    for k, v in tree_flatten(m.parameters()):
        if "lora" not in k:
            assert np.array_equal(np.asarray(v), frozen0[k]), k


@needs_mlx
def test_checkpoint_roundtrip_is_bit_identical(tmp_path):
    import mlx.core as mx
    import mlx.optimizers as optim

    def batch(m, step):
        ck = f"c{step}"
        s = [S("t", 0, [7, 9], 1, prompt=[1, 2], ckpt=ck, step=step),
             S("t", 1, [8], 0, prompt=[1, 2], ckpt=ck, step=step)]
        return _samples_with_lp(m, s)

    def run(m, opt, step):
        G.train_step(m, opt, batch(m, step), G.GRPOConfig(), policy_checkpoint_id=f"c{step}", policy_step=step)

    a = _lora_qwen3()
    oa = optim.Adam(learning_rate=1e-2)
    run(a, oa, 0)
    G.reseed(42, 1)
    expect_next = float(mx.random.uniform(shape=(1,)).item())
    run(a, oa, 1)

    b = _lora_qwen3()
    ob = optim.Adam(learning_rate=1e-2)
    run(b, ob, 0)
    man = G.save_checkpoint(tmp_path / "ck", b, ob, step=1, base_seed=42, extra={"data_cursor": 1})
    del b, ob

    c = _lora_qwen3(lora_seed=5)  # different LoRA init: must be fully overwritten by the checkpoint
    oc = optim.Adam(learning_rate=1e-2)
    mx.random.seed(999)
    got = G.load_checkpoint(tmp_path / "ck", c, oc)
    assert got["step"] == 1 and got["extra"] == {"data_cursor": 1} and got == man
    assert float(mx.random.uniform(shape=(1,)).item()) == expect_next  # RNG schedule restored
    assert int(oc.state["step"].item()) == 1
    run(c, oc, 1)
    assert G.trainable_digest(c) == G.trainable_digest(a)

    # tampering is detected
    f = tmp_path / "ck" / "optimizer.safetensors"
    raw = bytearray(f.read_bytes())
    raw[-1] ^= 0xFF
    f.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="hash mismatch"):
        G.load_checkpoint(tmp_path / "ck", _lora_qwen3(), optim.Adam(learning_rate=1e-2))


# ---------------------------------------------------------------- C1 Option A: exact-ratio sampling
@pytest.mark.parametrize("kwargs", [
    {"temperature": 0.7, "top_p": 0.95},
    {"temperature": 0.7, "top_p": 0.999999},
    {"temperature": 0.7, "top_p": 0.0},
    {"temperature": 0.7, "top_p": 1.0, "top_k": 50},
    {"temperature": 0.7, "top_p": 1.0, "min_p": 0.05},
    {"temperature": 0.0, "top_p": 1.0},
    {"temperature": math.nan, "top_p": 1.0},
    {"temperature": 0.7, "top_p": True},
])
def test_exact_ratio_objective_rejects_truncated_or_greedy_sampling(kwargs):
    with pytest.raises(G.TruncatedSamplingError):
        G.check_exact_ratio_sampling(**kwargs)


def test_exact_ratio_objective_accepts_untruncated_temperature_sampling():
    assert G.EXACT_RATIO_TOP_P == 1.0
    G.check_exact_ratio_sampling(temperature=0.7, top_p=1.0)
    G.check_exact_ratio_sampling(temperature=1, top_p=1, top_k=0, min_p=0)


def _load_vertical_slice():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts/grpo_vertical_slice.py"
    spec = importlib.util.spec_from_file_location("grpo_vertical_slice_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vertical_slice_refuses_top_p_below_one_with_exact_ratio_objective(tmp_path):
    vs = _load_vertical_slice()
    args = vs.parse(["--run-dir", str(tmp_path / "run"), "--cache", str(tmp_path / "cache")])
    assert args.top_p == 1.0  # default is the untruncated distribution
    with pytest.raises(G.TruncatedSamplingError):
        vs.Slice(vs.parse(["--run-dir", str(tmp_path / "run"), "--cache", str(tmp_path / "cache"),
                           "--top-p", "0.95"]))
    assert not (tmp_path / "run").exists()  # refused before creating the run
    with pytest.raises(G.TruncatedSamplingError):  # refused before touching a model
        vs.generate(None, None, [], seed=0, temperature=0.7, top_p=0.95, max_tokens=1)


@pytest.mark.skipif(not HAS_MLX, reason="mlx/mlx_lm not installed")
def test_top_p_one_sampler_draws_from_untruncated_temperature_softmax():
    # The mlx_lm sampler at top_p=1.0 applies no filter: it is categorical over
    # logprobs / T, i.e. softmax(logits / T), which token_logprobs models.
    import mlx.core as mx
    from mlx_lm.sample_utils import make_sampler

    logits = mx.array([[2.0, 0.5, -1.0, -3.0, 0.0, -0.25]])
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    t = 0.7
    sampler = make_sampler(temp=t, top_p=1.0, min_p=0.0, top_k=0)
    for seed in range(20):
        mx.random.seed(seed)
        got = int(sampler(logprobs).item())
        mx.random.seed(seed)
        want = int(mx.random.categorical(logprobs * (1 / t)).item())
        assert got == want
