"""Prospective publication/rollback control on newly declared synthetic prompts."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import psutil
from async_verifier import backend, restore, dump
from rvl_systems.hf_trainer import HFCausalLMGRPOTrainer


def main(protocol_path, output):
    q = json.loads(protocol_path.read_text())
    p = json.loads(Path(q["parent_protocol"]).read_text())
    assert not output.exists()
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == q["source_sha256"]
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory().used
    assert vm.available >= p["limits"]["min_available"]
    output.mkdir()
    torch, tok, model, ids = backend(p)
    trainer = HFCausalLMGRPOTrainer(model)
    started = time.monotonic()
    results = []

    def evaluate(cp, adverse=False):
        meta = restore(cp, model, trainer, torch)
        if adverse:
            with torch.no_grad():
                model.vare_action_bias[ids[2:]] -= 20
        values = []
        for text in q["evaluation_prompts"]:
            inputs = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
            )
            with torch.inference_mode():
                probs = torch.softmax(
                    model(torch.tensor([inputs])).logits[0, -1, ids], dim=0
                )
            values.append(float(probs[2:].sum()))
            assert psutil.swap_memory().used - swap <= p["limits"]["max_swap_growth"]
            assert (
                psutil.virtual_memory().available >= p["limits"]["hard_min_available"]
            )
            assert psutil.Process().memory_info().rss <= p["limits"]["max_rss"]
            assert time.monotonic() - started <= 120
        return {
            "checkpoint_sha256": meta["sha256"],
            "adverse_override": adverse,
            "score": sum(values) / len(values),
            "prompt_scores": values,
            "adapter_tensor_sha256": hashlib.sha256(
                model.vare_action_bias.detach().numpy().tobytes()
            ).hexdigest(),
        }

    for seed in p["seeds"]:
        root = Path(q["source_run"]) / f"seed-{seed}" / "D"
        inc = evaluate(root / "checkpoint-0.pt.gz")
        candidate = evaluate(root / "checkpoint-4.pt.gz")
        pointer = inc["checkpoint_sha256"]
        dump(output / f"active-{seed}.json", {"checkpoint_sha256": pointer})
        promote = candidate["score"] >= inc["score"]
        if promote:
            pointer = candidate["checkpoint_sha256"]
            dump(output / f"active-{seed}.json", {"checkpoint_sha256": pointer})
        bad = evaluate(root / "checkpoint-4.pt.gz", True)
        baseline_score = candidate["score"] if promote else inc["score"]
        reject = bad["score"] < baseline_score
        assert (
            reject
            and json.loads((output / f"active-{seed}.json").read_text())[
                "checkpoint_sha256"
            ]
            == pointer
        )
        results.append(
            {
                "seed": seed,
                "incumbent": inc,
                "candidate": candidate,
                "promoted": promote,
                "negative_control": bad,
                "negative_rejected": reject,
                "active_checkpoint_sha256": pointer,
            }
        )
    dump(
        output / "receipts.json",
        {
            "status": "PASS",
            "results": results,
            "swap_growth": max(0, psutil.swap_memory().used - swap),
            "wall_seconds": time.monotonic() - started,
            "rss_bytes": psutil.Process().memory_info().rss,
            "rule": q["rule"],
            "scope": "synthetic publication control only; no independent model capability improvement",
        },
    )


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--protocol", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    args = a.parse_args()
    main(args.protocol, args.output)
