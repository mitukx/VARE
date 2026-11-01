"""Opt-in operator-authorized interlock for the VARE outer control plane.

This is a checkpoint/fence, NOT a sandbox, OS kill switch, or replacement for
independent evaluation. The signed authorization file and pinned assets MUST be
kept in a privilege domain inaccessible to candidate processes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import hmac
import json
import math
import re
import time
from typing import Callable, Mapping

from .evidence import canonical_json, sha256_file


class SafetyInterlockDenied(RuntimeError):
    """A mandatory control-plane authorization or integrity check failed."""


_ACTIONS = frozenset({"rollout", "train", "evaluate", "promote"})
_DOMAIN = "vare.control-lease.v1"
_HEX256 = re.compile(r"[a-f0-9]{64}\Z")


def sign_control_lease(
    *,
    key: bytes,
    run_id: str,
    incumbent_id: str,
    expires_at: int,
    allowed_actions: tuple[str, ...],
    approved_candidates: tuple[str, ...] = (),
) -> str:
    """Trusted operator ONLY: return signed JSON, never expose key to agents.

    For promotion the operator must approve the specific candidate identity.
    A signed lease authorizes a phase but does not assert evaluator correctness.
    """
    if type(key) is not bytes or len(key) < 32:
        raise ValueError("control signing key must be at least 32 bytes")
    if (
        not isinstance(run_id, str) or not run_id.strip()
        or not isinstance(incumbent_id, str) or not incumbent_id.strip()
        or type(expires_at) is not int or expires_at <= 0
        or type(allowed_actions) is not tuple
        or any(action not in _ACTIONS for action in allowed_actions)
        or len(set(allowed_actions)) != len(allowed_actions)
        or type(approved_candidates) is not tuple
        or any(not isinstance(c, str) or not c for c in approved_candidates)
        or len(set(approved_candidates)) != len(approved_candidates)
    ):
        raise ValueError("invalid signed control lease fields")
    body = {
        "domain": _DOMAIN,
        "run_id": run_id,
        "incumbent_id": incumbent_id,
        "expires_at": expires_at,
        "allowed_actions": sorted(allowed_actions),
        "approved_candidates": sorted(approved_candidates),
    }
    mac = hmac.new(key, canonical_json(body).encode("utf-8"), hashlib.sha256).hexdigest()
    return canonical_json({**body, "mac_sha256": mac}) + "\n"


@dataclass(frozen=True, repr=False)
class SafetyInterlock:
    """Freshly re-read signed authority and frozen asset hashes on every phase.

    The authority file must not be writable by the candidate. Removing the
    file revokes future operations; this does not interrupt already running
    model calls or OS processes.
    """

    control_path: Path
    key: bytes
    run_id: str
    protected_assets: Mapping[Path, str] = field(default_factory=dict)
    clock: Callable[[], float] = field(default=time.time, repr=False)

    def __post_init__(self) -> None:
        if type(self.key) is not bytes or len(self.key) < 32:
            raise ValueError("control verification key must be at least 32 bytes")
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be nonempty")
        for path, digest in self.protected_assets.items():
            if not isinstance(path, Path) or not isinstance(digest, str) or _HEX256.fullmatch(digest) is None:
                raise ValueError("protected assets must map paths to lowercase SHA-256 digests")

    def require(self, action: str, *, incumbent_id: str, candidate_id: str | None = None) -> None:
        if action not in _ACTIONS:
            raise SafetyInterlockDenied("unknown_control_action")
        try:
            path = Path(self.control_path)
            # Refuse links, including parent links, and oversized/unbounded
            # control files. The actual read must use an OS-protected mount.
            if any(item.is_symlink() for item in (path, *path.parents)):
                raise ValueError("control symlink")
            with path.open("rb") as handle:
                encoded = handle.read(8193)
            if len(encoded) > 8192:
                raise ValueError("oversized control lease")

            def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
                result: dict[str, object] = {}
                for k, v in pairs:
                    if k in result:
                        raise ValueError("duplicate control field")
                    result[k] = v
                return result

            data = json.loads(encoded, object_pairs_hook=unique_pairs)
            keys = {"domain", "run_id", "incumbent_id", "expires_at",
                    "allowed_actions", "approved_candidates", "mac_sha256"}
            if type(data) is not dict or set(data) != keys:
                raise ValueError("invalid control lease schema")
            received = data.pop("mac_sha256")
            if not isinstance(received, str) or _HEX256.fullmatch(received) is None:
                raise ValueError("malformed signature")
            expected = hmac.new(
                self.key, canonical_json(data).encode("utf-8"), hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(received, expected):
                raise ValueError("control signature mismatch")
            if (
                data["domain"] != _DOMAIN
                or data["run_id"] != self.run_id
                or data["incumbent_id"] != incumbent_id
                or type(data["expires_at"]) is not int
                or not math.isfinite(self.clock())
                or self.clock() >= data["expires_at"]
                or type(data["allowed_actions"]) is not list
                or action not in data["allowed_actions"]
                or type(data["approved_candidates"]) is not list
            ):
                raise ValueError("control scope or expiry mismatch")
            if action == "promote" and (
                not isinstance(candidate_id, str)
                or candidate_id not in data["approved_candidates"]
            ):
                raise ValueError("candidate lacks promotion authorization")
            for asset, digest in self.protected_assets.items():
                # The digest is operator-pinned, not derived from candidate
                # metadata. Hashes are defense-in-depth, NOT an OS ACL.
                if any(item.is_symlink() for item in (asset, *asset.parents)):
                    raise ValueError("protected asset symlink")
                if not asset.is_file() or sha256_file(asset) != digest:
                    raise ValueError("protected evaluator asset modified")
        except (ValueError, TypeError, AttributeError, KeyError, OSError, UnicodeError,
                json.JSONDecodeError, OverflowError) as exc:
            raise SafetyInterlockDenied("control_authorization_or_integrity_failed") from exc
