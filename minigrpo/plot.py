"""Write an SVG of the JSONL training metrics. No plotting dependency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SERIES = (
    "reward_mean",
    "entropy_mean",
    "response_length_mean",
    "kl_mean",
    "format_compliance_rate",
)


def _rows(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _polyline(values: list[float], width: int, height: int) -> str:
    if not values:
        return ""
    low = min(values)
    high = max(values)
    span = high - low or 1.0
    points = []
    for index, value in enumerate(values):
        x = 40 if len(values) == 1 else 40 + (width - 50) * index / (len(values) - 1)
        y = 8 + (height - 16) * (1.0 - (value - low) / span)
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def render_svg(rows: list[dict[str, object]]) -> str:
    train = [row for row in rows if row.get("kind") == "train"]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="760" font-family="sans-serif">',
        '<rect width="720" height="760" fill="white"/>',
        '<text x="16" y="24" font-size="16">minigrpo step metrics</text>',
    ]
    for slot, name in enumerate(SERIES):
        top = 40 + slot * 140
        values = [float(row[name]) for row in train if isinstance(row.get(name), (int, float))]
        parts.append(f'<text x="16" y="{top + 16}" font-size="13">{name}</text>')
        parts.append(f'<rect x="32" y="{top + 24}" width="660" height="100" fill="none" stroke="#ccc"/>')
        points = _polyline(values, 660, 100)
        if points:
            parts.append(
                f'<polyline fill="none" stroke="#1f4e79" stroke-width="2" '
                f'transform="translate(0,{top + 24})" points="{points}"/>'
            )
    summary = [row for row in rows if row.get("kind") == "dev_summary"]
    if summary:
        last = summary[-1]
        parts.append(
            f'<text x="16" y="744" font-size="13">dev accuracy {last.get("accuracy")} '
            f'n={last.get("n")}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot minigrpo JSONL metrics to SVG")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    svg = render_svg(_rows(args.metrics))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
