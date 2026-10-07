"""Прогоняет все конфиги из configs/ и собирает таблицу абляций.

    python src/rag/run_ablation.py                 # все конфиги, пропуская уже посчитанные
    python src/rag/run_ablation.py --only 04 10    # только конфиги с этими префиксами
    python src/rag/run_ablation.py --force         # пересчитать всё
    python src/rag/run_ablation.py --table         # только пересобрать таблицу

Каждый конфиг запускается отдельным процессом: модель полностью выгружается из видеопамяти
перед следующей (важно для карт с 4 ГБ).
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "results" / "retrieval.csv"


def current_configs() -> list[Path]:
    """Только пронумерованные конфиги вида 01_имя.yaml."""
    return sorted((ROOT / "configs").glob("[0-9][0-9]_*.yaml"))


def latest_rows(split: str) -> dict[str, dict]:
    if not CSV.exists():
        return {}
    rows = {}
    with CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("split") == split:
                rows[r["name"]] = r  # последняя запись по имени побеждает
    return rows


def _fmt(v) -> str:
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "—"


def write_table(split: str) -> str:
    names = {c.stem for c in current_configs()}
    rows = {k: v for k, v in latest_rows(split).items() if k in names}
    if not rows:
        return "нет результатов"
    head = ("| Конфигурация | Фрагментов | Recall@5 | 95% ДИ | Recall@10 | MRR@10 | Recall@1500 слов | Задержка, мс | Видеопамять, МБ |\n"
            "|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for name in sorted(rows):
        r = rows[name]
        lines.append(f"| {name} | {r['n_chunks']} | {float(r['recall@5']):.3f} | {r['recall@5_ci95']} | "
                     f"{float(r['recall@10']):.3f} | {float(r['mrr@10']):.3f} | {_fmt(r.get('recall@1500w'))} | {r['latency_ms_median']} | "
                     f"{r.get('gpu_peak_mb', '')} |")
    n = next(iter(rows.values()))["n_questions"]
    text = f"# Абляции поиска ({split}, {n} отвечаемых вопросов)\n\n" + head + "\n".join(lines) + "\n"
    (ROOT / "results" / f"ablation_{split}.md").write_text(text, encoding="utf-8")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--only", nargs="*", default=None, help="префиксы имён конфигов, например 04 10")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()

    if not args.table:
        done = latest_rows(args.split)
        configs = current_configs()
        if args.only:
            configs = [c for c in configs if any(c.stem.startswith(p) for p in args.only)]
        for cfg in configs:
            if cfg.stem in done and not args.force:
                print(f"= {cfg.stem}: уже посчитан, пропускаю (--force чтобы пересчитать)")
                continue
            print(f"\n>>> {cfg.stem}", flush=True)
            res = subprocess.run([sys.executable, str(ROOT / "src/rag/eval_retrieval.py"), str(cfg),
                                  "--split", args.split])
            if res.returncode != 0:
                print(f"!!! {cfg.stem} упал (код {res.returncode}), продолжаю со следующим")
    print("\n" + write_table(args.split))


if __name__ == "__main__":
    main()
