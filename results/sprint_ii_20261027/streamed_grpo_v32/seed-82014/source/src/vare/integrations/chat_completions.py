from __future__ import annotations

import asyncio
import json
import time
import urllib.request
from dataclasses import dataclass

from ..types import Attempt, Task


@dataclass(slots=True)
class ChatCompletionsRollout:
    """Minimal HTTP chat-completions rollout adapter."""

    base_url: str
    model: str
    api_key: str | None = None
    timeout_s: float = 120.0
    temperature: float = 0.7
    max_tokens: int = 1024

    async def rollout(self, task: Task, policy_id: str, policy_version: int, step: int) -> Attempt:
        return await asyncio.to_thread(self._sync_rollout, task, policy_id, policy_version, step)

    def _sync_rollout(self, task: Task, policy_id: str, policy_version: int, step: int) -> Attempt:
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": task.prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url.rstrip("/") + "/v1/chat/completions", data=payload, headers=headers, method="POST")
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = json.loads(resp.read().decode())
        latency_ms = 1000 * (time.perf_counter() - t0)
        output = body["choices"][0]["message"]["content"]
        return Attempt(task=task, output=output, policy_id=policy_id, policy_version=policy_version, created_step=step, latency_ms=latency_ms, metadata={"backend": "chat-completions-http"})
