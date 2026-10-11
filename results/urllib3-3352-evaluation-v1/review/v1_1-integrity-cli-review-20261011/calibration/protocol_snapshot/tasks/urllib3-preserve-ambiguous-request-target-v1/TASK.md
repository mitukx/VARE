# Task: preserve ambiguous request-target paths

## Problem

At the pinned urllib3 source revision, `PoolManager().urlopen()` raises
`LocationParseError` for a request such as
`http://127.0.0.1:10000//v:h`. The authority is `127.0.0.1:10000`; `//v:h` is
the path. A colon is legal inside a URI path. A local HTTP server should receive
the request with the same path, including its leading slashes.

## Goal

Fix the production request path handling so an absolute URL with a path that
resembles a scheme-less authority is sent to the URL's actual host and port,
without changing the request-target path. Preserve ordinary request paths.

Keep the change limited to `src/urllib3/connectionpool.py`. Return a unified diff
for that file only. Do not add tests or modify task/evaluator files.
The accepted change is confined to the existing
`HTTPConnectionPool.urlopen` implementation. The grader checks that the tested
request traverses urllib3's PoolManager, connection-pool, connection, and
socket-creation path; direct socket requests and import-time behavior are not
accepted as task solutions.

## Reproduction

The protected evaluator starts a local HTTP server and invokes the pinned
package through `PoolManager().urlopen()`. It captures the actual HTTP request
line. The base revision should fail the `//v:h` case before a request is sent;
ordinary paths remain valid. No internet connection is used by the evaluator.

This is a development task only. The issue and its eventual fix are public; a
pass is not held-out model evidence.
