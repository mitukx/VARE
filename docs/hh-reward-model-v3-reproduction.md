# Reproduce the HH-RLHF v3 score audit

This guide describes the local replay for the outcome-informed HH-RLHF v3 fixed-head calibration study. It reruns context selection, frozen feature extraction, reward-head/scalar fitting, per-prompt scores, and the frozen gate. It does not train a language model or establish downstream policy improvement.

## Pinned inputs

- Repository commit: `b35232bab1c284713d8e65aaa4ee7a4e94fab405`
- Protocol: [`cpu_hh_reward_model_v3.lock.json`](../protocols/cpu_hh_reward_model_v3.lock.json), SHA-256 `2a891c33b7c8c016d27e77fd95eeb4c4557d2cc2b3f1ebe193416ad95e86b4a4`
- Dataset: `Anthropic/hh-rlhf` revision `09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa`, `helpful-base/train.jsonl.gz` and `helpful-base/test.jsonl.gz`
- Model: `Qwen/Qwen2.5-0.5B-Instruct` revision `7ae557604adf67be50417f59c2c2f167def9a775`
- Runtime: Python 3.9.6, PyTorch 2.8.0, Transformers 4.57.3, Datasets 4.4.2, NumPy 1.26.4
- Resources: CPU only, four PyTorch threads, at most 6 GiB RSS and one hour. The auditor disables network access.

The model and dataset are public pinned Hub snapshots. Hub's official CLI supports downloading files from a specific repository revision; see the [Hugging Face CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli). Run the setup below while online, then run the audit offline. The scripts expect the default Hugging Face cache under `~/.cache/huggingface/hub`.

```bash
python3.9 -m venv .venv-hh-v3
source .venv-hh-v3/bin/activate
python -m pip install torch==2.8.0 transformers==4.57.3 datasets==4.4.2 numpy==1.26.4 huggingface_hub

hf download Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  config.json model.safetensors tokenizer.json tokenizer_config.json vocab.json merges.txt

hf download Anthropic/hh-rlhf --repo-type dataset \
  --revision 09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa \
  helpful-base/train.jsonl.gz helpful-base/test.jsonl.gz
```

Verify the repository revision before opening any files, then run the audit from the checkout root:

```bash
git rev-parse HEAD
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/audit_cpu_hh_reward_model_v3.py \
  results/cpu-hh-reward-model-v3/confirmation/run-1
```

The expected final JSON has `status: "pass"`, 306 prompt contexts, mean calibrated-minus-raw NLL `-0.174543534035452`, and interval `[-0.233650734523746, -0.11775793077963356]`. The committed `audit.json` SHA-256 is `e5df647f531ae5ea247b885c5b5b55e6cf9a21c2fa20e28247f22819e6ba90b5`. A mismatch is a failed reproduction and should be investigated before citing the result.

## Scope of the replay

The confirmation bundle binds every included file by a SHA-256 manifest. The auditor verifies that manifest, protocol and code snapshots, historical context exclusions, fresh train/test row selection, input model/data hashes, independently refit scores, metrics, and the prompt-bootstrap decision. Confirmation test rows are read because this command replays the already-completed confirmation; do not use it as a development command or change the frozen protocol.

An author-run depth-1 clean clone of commit `b35232bab1c284713d8e65aaa4ee7a4e94fab405` passed this audit, and the generated `audit.json` was byte-identical to the committed artifact. This is same-author, same-host reproduction using the same model, tokenizer, and runtime. It is not external human reproduction, a different implementation of the model, or evidence of downstream task/RL improvement. The result remains outcome-informed: its practical threshold was informed by the already-opened v2 post-hoc analysis. See the [confirmation report](hh-reward-model-v3-confirmation-report.md) for complete metrics and limits.
