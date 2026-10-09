#!/usr/bin/env python3
"""Generate a deterministic, procedurally verifiable multi-step arithmetic task set."""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/cpu-qwen-math-tir-feasibility-v1/pilot.jsonl"
SEED = 20261009
PER_FAMILY = 24


def generate_rows():
    rng = random.Random(SEED)
    rows = []
    for i in range(PER_FAMILY):
        initial = rng.randint(120, 640)
        pallets = rng.randint(3, 28)
        per_pallet = rng.randint(9, 46)
        damaged = rng.randint(2, 35)
        shipped = rng.randint(25, 240)
        answer = initial + pallets * per_pallet - damaged - shipped
        rows.append({"task_id": f"inventory-{i:03d}", "family": "inventory",
            "prompt": f"A warehouse had {initial} boxes. It received {pallets} pallets with {per_pallet} boxes each, removed {damaged} damaged boxes, and shipped {shipped} boxes. How many boxes remain?",
            "gold_answer": answer})

        adults = rng.randint(18, 180)
        adult_price = rng.randint(12, 55)
        students = rng.randint(20, 240)
        student_price = rng.randint(5, adult_price - 1)
        refund = rng.randint(10, 900)
        answer = adults * adult_price + students * student_price - refund
        rows.append({"task_id": f"event-{i:03d}", "family": "event_revenue",
            "prompt": f"At a school event, {adults} adult tickets were sold for {adult_price} dollars each, and {students} student tickets were sold for {student_price} dollars each. The organizer later refunded {refund} dollars. How many dollars were kept?",
            "gold_answer": answer})

        machines = rng.randint(3, 18)
        hourly = rng.randint(14, 95)
        hours = rng.randint(3, 16)
        rejected = rng.randint(4, machines * hourly * hours // 5)
        answer = machines * hourly * hours - rejected
        rows.append({"task_id": f"factory-{i:03d}", "family": "factory_output",
            "prompt": f"A factory has {machines} machines. Each machine makes {hourly} parts per hour. They run for {hours} hours, and quality control rejects {rejected} parts. How many good parts remain?",
            "gold_answer": answer})

        cartons = rng.randint(8, 35)
        books = rng.randint(14, 48)
        classes = rng.randint(4, 12)
        loose = (rng.randint(0, classes - 1) - cartons * books) % classes
        donation = rng.randint(1, (cartons * books + loose) // classes - 1) * classes
        total = cartons * books + loose
        answer = (total - donation) // classes
        rows.append({"task_id": f"classroom-{i:03d}", "family": "equal_distribution",
            "prompt": f"A school has {cartons} cartons with {books} books each, plus {loose} loose books. It donates {donation} books. The remaining books are shared equally among {classes} classes. How many books does each class receive?",
            "gold_answer": answer})
    return rows


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_rows()
    if len(rows) != 4 * PER_FAMILY or len({r['task_id'] for r in rows}) != len(rows):
        raise ValueError("generated task inventory is incomplete or duplicated")
    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"rows": len(rows), "families": 4, "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
