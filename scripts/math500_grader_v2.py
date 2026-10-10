"""Independent exact-answer evaluator for the frozen MATH-500 study.

Training reward is supplied by TRL's ``accuracy_reward``. This evaluator keeps
its own final-box extraction and SymPy equality path so held-out task success
is not reconstructed from the training reward values.
"""

from __future__ import annotations

import re
import signal
import threading
import time
from dataclasses import dataclass
from typing import Any

import sympy
from latex2sympy2_extended import latex2sympy
from sympy import MatrixBase, Set, Tuple
from sympy.core.relational import Relational


MAX_COMPLETION_CHARS = 131_072
MAX_FINAL_BOX_CHARS = 8_192
PARSER_TIMEOUT_SECONDS = 3


class _ParserTimeout(TimeoutError):
    pass


def _with_symbolic_timeout(operation):
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
        return operation()
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def timeout(_signum, _frame):
        raise _ParserTimeout("symbolic parsing exceeded the per-answer time limit")

    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, PARSER_TIMEOUT_SECONDS)
    try:
        return operation()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(
                signal.ITIMER_REAL,
                max(0.001, previous_timer[0] - (time.monotonic() - started)),
                previous_timer[1],
            )


def _latex2sympy_with_timeout(text: str):
    return _with_symbolic_timeout(lambda: latex2sympy(text))


@dataclass(frozen=True)
class ParsedAnswer:
    raw: str
    value: Any | None
    fallback: str


def _last_boxed(text: str) -> str | None:
    """Return the contents of the last balanced ``\\boxed{...}`` expression."""
    if len(text) > MAX_COMPLETION_CHARS:
        return None
    starts = [m.start() for m in re.finditer(r"\\boxed\s*\{", text)]
    if not starts:
        return None

    # Inspect only the final box marker. Falling back to an earlier valid box
    # would let a malformed final answer inherit correctness from scratch work.
    brace = text.find("{", starts[-1])
    depth = 0
    for index in range(brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                answer = text[brace + 1 : index].strip()
                return answer if answer and len(answer) <= MAX_FINAL_BOX_CHARS else None
    return None


def _normalize_units(answer: str) -> str:
    value = answer.strip()
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    value = re.sub(r"\\(?:text|mbox)\s*\{\s*(?:degrees?|deg)\s*\}", r"\\degree", value, flags=re.I)
    value = re.sub(r"\^\s*\\(?:circ|degree)\b|\\(?:circ|degree)\b", r"*\\pi/180", value)
    value = re.sub(r"\\\\%|%", r"/100", value)
    value = value.replace(r"\$", "")
    return value


def _fallback_key(answer: str) -> str:
    value = answer.strip().lower()
    value = re.sub(r"\\(?:text|mbox)\s*\{([^{}]*)\}", r"\1", value)
    value = value.replace(r"\$", "")
    value = re.sub(r"\\(?:left|right)\b", "", value)
    value = re.sub(r"\s+", "", value)
    return value.strip(".$,; ")


def _textual_value(answer: str) -> str | None:
    value = answer.strip()
    match = re.fullmatch(r"\\(?:text|mbox)\s*\{([^{}]+)\}", value, flags=re.I)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip().casefold()
    if re.fullmatch(r"[A-Za-z][A-Za-z .'-]*", value):
        return re.sub(r"\s+", " ", value).strip().casefold()
    return None


def _parse_expression(answer: str) -> Any | None:
    normalized = _normalize_units(answer)
    if not normalized:
        return None
    try:
        return _latex2sympy_with_timeout(normalized)
    except Exception:
        return None


def parse_prediction(text: str) -> ParsedAnswer | None:
    boxed = _last_boxed(text)
    if boxed is None:
        return None
    return ParsedAnswer(boxed, _parse_expression(boxed), _fallback_key(boxed))


def parse_reference(answer: str) -> ParsedAnswer:
    reference = answer.strip()
    return ParsedAnswer(reference, _parse_expression(reference), _fallback_key(reference))


def _same_value(reference: Any, prediction: Any) -> bool:
    if reference == prediction:
        return True

    if isinstance(reference, MatrixBase) and isinstance(prediction, Tuple):
        if reference.rows == 1 or reference.cols == 1:
            values = tuple(reference)
            return len(values) == len(prediction) and all(
                _same_value(a, b) for a, b in zip(values, prediction)
            )
        return False
    if isinstance(prediction, MatrixBase) and isinstance(reference, Tuple):
        return _same_value(prediction, reference)

    if isinstance(reference, MatrixBase) or isinstance(prediction, MatrixBase):
        if not (isinstance(reference, MatrixBase) and isinstance(prediction, MatrixBase)):
            return False
        if reference.shape != prediction.shape:
            return False
        return all(sympy.simplify(a - b) == 0 for a, b in zip(reference, prediction))

    if isinstance(reference, Tuple) or isinstance(prediction, Tuple):
        if not (isinstance(reference, Tuple) and isinstance(prediction, Tuple)):
            return False
        return len(reference) == len(prediction) and all(
            _same_value(a, b) for a, b in zip(reference, prediction)
        )

    if isinstance(reference, Set) or isinstance(prediction, Set):
        return isinstance(reference, Set) and isinstance(prediction, Set) and reference == prediction

    if isinstance(reference, Relational) or isinstance(prediction, Relational):
        return isinstance(reference, Relational) and isinstance(prediction, Relational) and sympy.simplify(reference) == sympy.simplify(prediction)

    try:
        difference = sympy.simplify(sympy.together(reference - prediction))
        return difference == 0 or difference is sympy.S.Zero
    except (TypeError, ValueError, AttributeError):
        return sympy.simplify(reference) == sympy.simplify(prediction)


def exact_match(prediction_text: str, reference_text: str) -> bool:
    """Score one completion; missing/unbalanced final boxes fail closed."""
    try:
        prediction = parse_prediction(prediction_text)
        if prediction is None:
            return False
        reference = parse_reference(reference_text)
        if r"\text{" in reference.raw or r"\mbox{" in reference.raw:
            predicted_text = _textual_value(prediction.raw)
            reference_text_value = _textual_value(reference.raw)
            if predicted_text is not None and reference_text_value is not None:
                return predicted_text == reference_text_value
        if prediction.value is not None and reference.value is not None:
            return bool(_with_symbolic_timeout(lambda: _same_value(reference.value, prediction.value)))
        # Text fallback is reserved for explicitly textual answer forms. An
        # unparseable mathematical expression must not pass by string identity.
        predicted_text = _textual_value(prediction.raw)
        reference_text_value = _textual_value(reference.raw)
        return (
            predicted_text is not None
            and reference_text_value is not None
            and predicted_text == reference_text_value
        )
    except Exception:
        # Generated strings are untrusted input. Parser/simplifier failures are
        # grading failures, never positive reward or evaluator crashes.
        return False


def has_valid_final_box(text: str) -> bool:
    return parse_prediction(text) is not None
