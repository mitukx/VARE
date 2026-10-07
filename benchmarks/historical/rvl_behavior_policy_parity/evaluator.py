"""External regression evaluator for a historical sampler/learner parity bug.

This grader intentionally lives outside the candidate repository. It checks the
minimal source-level contract that the original failure violated. It is a first
historical environment seed, not a substitute for the full numerical test suite.
"""
from __future__ import annotations

from pathlib import Path
import sys


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main(root: str) -> None:
    repo = Path(root)
    backend = (repo / "src/rvl_systems/hf_backend.py").read_text(encoding="utf-8")
    trainer = (repo / "src/rvl_systems/hf_trainer.py").read_text(encoding="utf-8")

    require('"sampling_temperature": temperature' in backend, "rollout provenance must retain sampling temperature")
    require("GenerationConfig(" in backend, "rollout must start from an explicit neutral generation configuration")
    require('"top_k": 0' in backend and '"top_p": 1.0' in backend, "rollout sampling transform must be explicit")
    require("sampling_temperature" in trainer, "learner must consume behavior sampling temperature")
    require("response_logits.float() / temperature" in trainer, "learner log-probs must match the temperature-scaled behavior policy")
    require("greedy scores are not behavior probabilities" in trainer, "greedy rollout must fail closed for trainable behavior ratios")
    print("historical parity regression contract: PASS")


if __name__ == "__main__":
    main(sys.argv[1])
