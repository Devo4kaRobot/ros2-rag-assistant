"""Парное сравнение двух конфигураций на одних и тех же вопросах.

    python src/rag/compare_runs.py 05_e5base_recursive 06_e5base_structural [--metric r5] [--split dev]

Парный bootstrap: ресэмплируем вопросы и смотрим распределение разницы метрики. Если 95% ДИ
разницы содержит 0 — на этом наборе различие не доказано (честная формулировка для README).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load(name: str, split: str) -> dict[str, dict]:
    p = ROOT / "results" / "runs" / f"{name}_{split}.jsonl"
    return {r["id"]: r for r in map(json.loads, p.open(encoding="utf-8"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--metric", default="r5", choices=["r5", "r10", "mrr", "r1500w"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=10000)
    args = ap.parse_args()
    A, B = load(args.a, args.split), load(args.b, args.split)
    ids = sorted(set(A) & set(B))
    a = np.array([A[i][args.metric] for i in ids], dtype=float)
    b = np.array([B[i][args.metric] for i in ids], dtype=float)
    d = b - a
    rng = np.random.default_rng(0)
    boot = rng.choice(d, size=(args.n, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_two = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    print(f"{args.metric}: {args.a} = {a.mean():.3f}, {args.b} = {b.mean():.3f}, n = {len(ids)}")
    print(f"разница (B − A) = {d.mean():+.3f}, 95% ДИ [{lo:+.3f}; {hi:+.3f}], p ≈ {p_two:.3f}")
    print(f"B лучше на {int((d > 0).sum())} вопросах, хуже на {int((d < 0).sum())}, одинаково на {int((d == 0).sum())}")
    print("→ различие значимо" if lo > 0 or hi < 0 else "→ на этом наборе различие не доказано (ДИ содержит 0)")


if __name__ == "__main__":
    main()
