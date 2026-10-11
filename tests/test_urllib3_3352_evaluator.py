from __future__ import annotations

import importlib.util
from pathlib import Path
import socket
import threading
import time


EVALUATOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/historical/urllib3_3352/evaluator/grade.py"
)
SPEC = importlib.util.spec_from_file_location("urllib3_3352_grade", EVALUATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def test_request_line_reader_obeys_absolute_deadline_under_slow_drip() -> None:
    reader, writer = socket.socketpair()

    def dribble() -> None:
        try:
            for value in b"GET /slow HTTP/1.1\r\n":
                writer.send(bytes([value]))
                time.sleep(0.04)
        except OSError:
            pass
        finally:
            writer.close()

    sender = threading.Thread(target=dribble, daemon=True)
    sender.start()
    started = time.monotonic()
    try:
        line, error = EVALUATOR.read_request_line(reader, started + 0.12)
    finally:
        reader.close()
    elapsed = time.monotonic() - started
    sender.join(timeout=0.2)

    assert line is None
    assert error == "request-line deadline expired"
    assert elapsed < 0.3
