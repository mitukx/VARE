"""Single-event-loop coordination for trusted hooks sharing a mutable runtime."""
from __future__ import annotations

import asyncio
import weakref

from .promotion_journal import PromotionRecoveryRequired


class RuntimeCoordinator:
    """Serialize full rounds and retain uncertainty across loop instances.

    Distinct hook proxies for one runtime must explicitly share this object,
    or use the same durable PromotionJournal. This is not a security boundary.
    """

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.failure: BaseException | None = None

    def assert_usable(self) -> None:
        if self.failure is not None:
            raise PromotionRecoveryRequired(
                "unresolved runtime recovery required; quiesce all owners and "
                "independently verify restoration before operator recovery"
            ) from self.failure

    def quarantine(self, failure: BaseException) -> None:
        # A traceback can retain hooks/loops and defeat the weak registry's
        # lifecycle. Preserve diagnostics without retaining those resources.
        self.failure = RuntimeError(f"{type(failure).__name__}: {failure}")

    def recover(self) -> None:
        """Operator only: old work must be terminal and runtime already verified.

        Reconcile/recover the durable journal first when configured. This call
        cannot inspect weights, terminate workers, or prove external recovery.
        """
        if self.lock.locked():
            raise PromotionRecoveryRequired("cannot recover a live runtime round")
        self.failure = None


# Identity rather than object hashing supports unhashable dataclass hooks. A
# weak reference keeps the fence alive while hooks live, even after every loop
# is dropped. Non-weakrefable hooks are retained for process lifetime.
_COORDINATORS: dict[int, tuple[object, RuntimeCoordinator]] = {}


def coordinator_for(hooks: object) -> RuntimeCoordinator:
    key = id(hooks)
    existing = _COORDINATORS.get(key)
    if existing is not None:
        return existing[1]
    coordinator = RuntimeCoordinator()
    try:
        reference: object = weakref.ref(hooks, lambda _ref: _COORDINATORS.pop(key, None))
    except TypeError:
        reference = hooks
    _COORDINATORS[key] = (reference, coordinator)
    return coordinator
