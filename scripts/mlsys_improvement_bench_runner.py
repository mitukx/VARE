"""In-sandbox benchmark executor for the MLSys improvement loop.

Executed *inside* the sandbox as ``python -I -c <source> <spec>`` by
``scripts/mlsys_improvement_loop.py``; it must stay self-contained (standard
library only) and is never imported by the parent.

Order of operations (every security-relevant step precedes candidate import):

1. read one JSON line from standard input with the per-run frame key and the
   per-run input seed, then close standard input; neither is in argv, the
   environment or a file;
2. move the frame channel off file descriptor 1 (as the v2 test runner does);
3. execute the grader-owned workload source (passed in the spec) and build all
   inputs from the seed, before the candidate module exists in the process;
4. emit an authenticated ``ready`` frame, import the module and run every
   case, checking that each result has exact built-in types (so no work can be
   deferred past the timed window through lazy objects);
5. emit ``done``; only then hash the outputs and emit ``result``.

The parent measures the interval between receiving ``ready`` and ``done`` on
its own clock; the in-process duration in ``done`` is a diagnostic only.
Frames use the v2 format ``VARE_FRAME <hmac-sha256 hex> <json>``.
"""
import json
import os
import sys
import time


def _main():
    spec = json.loads(sys.argv[1])
    secret = json.loads(sys.stdin.readline())
    key = bytes.fromhex(secret["key"])
    seed = int(secret["seed"])
    if len(key) < 16:
        os._exit(97)
    sys.stdin.close()
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    channel = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    # Everything used after the candidate import is bound now, to C-level
    # callables where possible. Frames are formatted by hand (no json module)
    # and authenticated with _hashlib.hmac_digest, so rebinding attributes of
    # json, hmac or hashlib from candidate code cannot alter or forge them.
    from _hashlib import hmac_digest

    write, clock, type_, repr_ = os.write, time.perf_counter, type, repr
    str_, bool_, none_ = str, bool, type(None)
    nonce = str(int(spec["nonce"]))
    seq = [0]

    def emit(kind, fields):
        # fields: list of (name, already-encoded JSON value), names sorted.
        parts = fields + [("kind", '"' + kind + '"'), ("nonce", nonce), ("seq", repr_(seq[0]))]
        parts.sort()
        payload = "{" + ",".join('"' + k + '":' + v for k, v in parts) + "}"
        mac = hmac_digest(key, payload.encode(), "sha256").hex()
        data = ("VARE_FRAME " + mac + " " + payload + "\n").encode()
        while data:
            data = data[write(channel, data):]
        seq[0] += 1

    workload = {"__name__": "vare_workload"}
    exec(compile(spec["workload_source"], "<workload>", "exec"), workload)
    cases = workload["make_cases"](seed, int(spec["count"]), spec["profile"])
    digest = workload["outcome_digest"]
    for root in reversed(spec["import_roots"]):
        sys.path.insert(0, root)
    emit("start", [("cases", repr_(len(cases)))])

    outcomes = []
    status = "ok"
    emit("ready", [])
    t0 = clock()
    try:
        module = __import__(spec["module"])
        parse, apply_ = module.parse_blocks, module.apply_blocks
        for original, completion in cases:
            o = apply_(original, parse(completion))
            ok, reason, new_text = o.ok, o.reason, o.new_text
            if type_(ok) is not bool_ or type_(reason) is not str_ or (
                    type_(new_text) is not str_ and type_(new_text) is not none_):
                status = "bad_output_type"
                break
            outcomes.append((ok, reason, new_text))
    except BaseException as exc:  # noqa: B902 - any failure, incl. SystemExit, is reported
        status = "exception"
        del exc
    elapsed = clock() - t0
    emit("done", [("inproc_seconds", repr_(elapsed)), ("n", repr_(len(outcomes))),
                  ("status", '"' + status + '"')])
    result = digest(outcomes) if status == "ok" else ""
    emit("result", [("digest", '"' + result + '"')])
    sys.stderr.flush()
    os._exit(0 if status == "ok" else 1)


if __name__ == "__main__":
    _main()
