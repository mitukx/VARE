# Research decision log

This file is append-only. New decisions are appended with dated evidence and falsification criteria; prior entries are not rewritten after outcomes are observed.

## 2026-10-31 — Preserve frozen tool allowlists while the runtime grows

- **Hypothesis:** The v1/v2 Development campaign contracts must remain exactly as frozen; a later runtime tool must not become authorized for those campaigns implicitly.
- **Evidence:** CI failed because its tests required the locked v1/v2 allowlists to equal the runtime's expanded `tools_for("unified_diff")` result. The only additional runtime tool is `replace_text`. Both campaign protocols declare inference unauthorized, and their byte/hash locks are unchanged. The tool itself is constrained by existing exact-context, allowlisted-path tests.
- **Alternative explanations:** The frozen protocol should be edited and re-locked; or the runtime addition should be removed. Editing it would retrospectively change the experiment contract, while removing it would discard an intentional Stage 3 controller capability.
- **Decision:** Preserve both protocol files and hashes. Test that each frozen allowlist equals the current runtime schema with only the later `replace_text` addition excluded. A future campaign that wants this tool needs a separately versioned, frozen protocol before inference.
- **Falsification criteria:** Any other runtime tool addition or removal must make the contract test fail until explicitly versioned; a campaign allowlist containing a tool unavailable to its runner or including an unapproved edit tool invalidates this decision.
- **Follow-up:** After PR #150 CI passes, integrate the correction and version any new campaign schema prospectively before using it.

## 2026-10-31 — Reclassify the deep-qualification Development split after exposure audit

- **Hypothesis:** The three source-qualified Development tasks in deep-qualification v1 can be treated as unexposed because that qualification run itself started no model rollouts.
- **Evidence:** The qualification catalog is pinned at SHA-256 `2b2a1a7c6e64a866dd3732fcd9723999969a1ac69d86f8a4d9f35edf67322c9d` and labels all three tasks unseen. The retained Stage 2/3 exposure ledger at SHA-256 `a2da858454745c86ef7e5162d47552bb659943e03a47442cd03c0e9dff77693a` matches each by exact task ID, source PR, and task family: #45852 has one Stage 2 model attempt, #1912 has two Stage 2 and three Stage 3 attempts, and #1971 has two Stage 2 attempts. The eight attempts yielded no independently verified repair; #1912's applied edit failed 0/14 hidden cases. The frozen source protocol assigns these same three families to Development.
- **Alternative explanations:** Count only successful repairs as exposure; or leave the original “unseen” flags unchanged because the deep-qualification protocol did not itself call a model. Both would ignore that model prompts and task workspaces had already been presented in earlier campaigns.
- **Decision:** Preserve the original source protocol, catalog, and raw evidence byte-for-byte. Append an exposure disposition that marks all three Development tasks previously exposed and makes this split ineligible for fresh Development evaluation or reallocation to Train. No held-out task was used.
- **Falsification criteria:** The pinned task identity, source PR, family, Stage 2/3 attempt totals, or source hashes fail to match the retained records.
- **Follow-up:** Audit the manual candidate queue against the full exposure history, select a replacement Development pool from unexposed families, and independently qualify its baseline, known fix, wrong mutants, and grader boundary before any model run.

## 2026-10-31 — Classify a mismatched worker nonce as a protocol failure

- **Hypothesis:** A syntactically valid frame with the wrong nonce is a protocol mismatch and must not yield candidate data.
- **Evidence:** `_parse_result` compares the frame nonce before parsing/accepting its payload and returns `(None, "protocol_mismatch")`. The isolated-grader transport taxonomy includes this classification, and the existing transport regression test checks nonce mismatch.
- **Alternative explanations:** Keep the older test label `malformed_result`; this would collapse a well-formed but misbound frame into a syntax error and disagree with the transport taxonomy.
- **Decision:** Update the expected classification and assert the returned value is `None`; keep fail-closed nonce validation unchanged.
- **Falsification criteria:** A mismatched nonce is ever accepted, a payload is returned on mismatch, or the independent calibration no longer distinguishes protocol mismatch from malformed framing.
- **Follow-up:** Run the full Python CI matrix after pushing PR #150 and preserve its receipts.
