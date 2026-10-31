"""SEARCH/REPLACE edit format shared by every arm.

The model returns one or more blocks::

    <<<<<<< SEARCH
    exact lines from the current file
    =======
    replacement lines
    >>>>>>> REPLACE

Every SEARCH text must occur exactly once in the target file. Any parse or
apply failure is a typed failure, never a pass. Reference fixes for SFT are
rendered in the same format from the reverse of the bug-introducing patch.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

_BLOCK = re.compile(
    r"<{5,9} ?SEARCH[^\n]*\n(.*?)\n?={5,9}[^\n]*\n(.*?)\n?>{5,9} ?REPLACE", re.DOTALL
)


@dataclass(frozen=True)
class Block:
    search: str
    replace: str


@dataclass(frozen=True)
class ApplyOutcome:
    ok: bool
    reason: str
    new_text: str | None = None


def parse_blocks(text: str) -> list[Block]:
    # Drop an optional reasoning section some chat templates emit.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return [Block(m.group(1), m.group(2)) for m in _BLOCK.finditer(text)]


def apply_blocks(original: str, blocks: list[Block]) -> ApplyOutcome:
    if not blocks:
        return ApplyOutcome(False, "no_edit_block")
    text = original
    for i, block in enumerate(blocks):
        if not block.search.strip():
            return ApplyOutcome(False, f"empty_search_block_{i}")
        count = text.count(block.search)
        if count == 1:
            text = text.replace(block.search, block.replace, 1)
            continue
        if count > 1:
            return ApplyOutcome(False, f"ambiguous_search_block_{i}")
        # Tolerate trailing-whitespace differences only.
        relaxed = _relaxed_replace(text, block)
        if relaxed is None:
            return ApplyOutcome(False, f"search_not_found_block_{i}")
        text = relaxed
    if text == original:
        return ApplyOutcome(False, "no_op_edit")
    return ApplyOutcome(True, "applied", text)


def _relaxed_replace(text: str, block: Block) -> str | None:
    lines = text.split("\n")
    needle = [ln.rstrip() for ln in block.search.split("\n")]
    stripped = [ln.rstrip() for ln in lines]
    hits = [
        i for i in range(len(lines) - len(needle) + 1) if stripped[i : i + len(needle)] == needle
    ]
    if len(hits) != 1:
        return None
    i = hits[0]
    return "\n".join(lines[:i] + block.replace.split("\n") + lines[i + len(needle) :])


def render_blocks(blocks: list[Block]) -> str:
    out = []
    for b in blocks:
        out.append(f"<<<<<<< SEARCH\n{b.search}\n=======\n{b.replace}\n>>>>>>> REPLACE")
    return "\n".join(out)


@dataclass(frozen=True)
class FilePatch:
    path: str
    hunks: tuple[tuple[tuple[str, str], ...], ...]  # per hunk: (tag, line)


def parse_single_file_patch(patch: str) -> FilePatch:
    """Parse a one-file unified diff (as stored by SWE-smith)."""
    if patch.count("diff --git ") != 1:
        raise ValueError("patch must touch exactly one file")
    path = None
    hunks: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] | None = None
    for line in patch.split("\n"):
        if line.startswith("diff --git "):
            continue
        if line.startswith("+++ "):
            target = line[4:].strip()
            if target == "/dev/null":
                raise ValueError("file deletion is not supported")
            path = target[2:] if target.startswith("b/") else target
            continue
        if line.startswith("--- "):
            if line[4:].strip() == "/dev/null":
                raise ValueError("file creation is not supported")
            continue
        if line.startswith(("index ", "new file", "deleted file", "similarity", "rename")):
            if line.startswith(("new file", "deleted file", "rename")):
                raise ValueError("only in-place modifications are supported")
            continue
        if line.startswith("@@"):
            current = []
            hunks.append(current)
            continue
        if current is None:
            continue
        if line.startswith("\\"):
            continue
        if line == "":
            current.append((" ", ""))
            continue
        tag, body = line[0], line[1:]
        if tag not in " +-":
            raise ValueError(f"unexpected diff line: {line[:40]!r}")
        current.append((tag, body))
    if path is None or not hunks:
        raise ValueError("no file or hunks in patch")
    # A trailing empty context line is usually the patch terminator.
    for h in hunks:
        while h and h[-1] == (" ", ""):
            h.pop()
    return FilePatch(path, tuple(tuple(h) for h in hunks))


def reference_fix_blocks(patch: str) -> tuple[str, list[Block]]:
    """Blocks that undo the bug patch: SEARCH = buggy text, REPLACE = original."""
    fp = parse_single_file_patch(patch)
    blocks = []
    for hunk in fp.hunks:
        buggy = [body for tag, body in hunk if tag in " +"]
        fixed = [body for tag, body in hunk if tag in " -"]
        blocks.append(Block("\n".join(buggy), "\n".join(fixed)))
    return fp.path, blocks
