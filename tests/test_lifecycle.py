import asyncio
from vare.lifecycle import FailureDrivenTaskGenerator, VerifierRefreshController
from vare.types import FailureCluster, Task


class G:
    async def generate(self, failures, experiences, limit):
        return [Task("generated", "hard case", "hard")][:limit]


class R:
    async def maybe_refresh(self, failures, experiences):
        return bool(failures)


def test_lifecycle_protocol_shapes_are_executable():
    f = [FailureCluster("capability_failure", 1, ("hard",), 0.0)]
    tasks = asyncio.run(G().generate(f, [], 1))
    assert tasks[0].family == "hard"
    assert asyncio.run(R().maybe_refresh(f, []))
