"""Шаг 1 пайплайна: rst → data/processed/pages.jsonl.

Запуск:  python -m rag.build_corpus  (из папки src)  или  python src/rag/build_corpus.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rag.rst_parser import load_substitutions, parse_page  # noqa: E402

DOCS = ROOT / "data" / "raw" / "ros2_documentation"
# коммит ветки lyrical, по которому размечен тестовый набор (eval/questions.jsonl)
DOCS_COMMIT = "08601b3c41cf4be349eae5774bc92a9594788641"
OUT = ROOT / "data" / "processed" / "pages.jsonl"

# что исключаем и почему (решение задокументировано в README)
EXCLUDE = [
    re.compile(r"^Releases/.*Complete-Changelog$"),  # автосгенерированные списки PR, 20–35 тыс. слов каждый
    re.compile(r"(^|/)_"),                            # служебные и включаемые фрагменты (_*.rst)
]


def _warn_if_other_commit() -> None:
    import subprocess
    try:
        cur = subprocess.run(["git", "-C", str(DOCS), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return
    if cur and cur != DOCS_COMMIT:
        print(f"[!] документация на коммите {cur[:10]}, а тестовый набор размечен по {DOCS_COMMIT[:10]}: "
              "якоря разделов могут не совпасть")


def main() -> None:
    src = DOCS / "source"
    if not src.exists():
        sys.exit(f"Нет {src}. Скачай документацию ROS 2 (коммит {DOCS_COMMIT}) — см. «Запуск» в README")
    _warn_if_other_commit()
    subs = load_substitutions(DOCS / "global_substitutions.txt")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_pages = n_sections = n_words = 0
    skipped = []
    with OUT.open("w", encoding="utf-8") as f:
        for p in sorted(src.rglob("*.rst")):
            pid = p.relative_to(src).with_suffix("").as_posix()
            if any(rx.search(pid) for rx in EXCLUDE):
                skipped.append(pid)
                continue
            page = parse_page(p, src, subs)
            words = sum(len(s["text"].split()) for s in page["sections"])
            if words < 30:  # страницы-оглавления из одного toctree
                skipped.append(pid)
                continue
            page["n_words"] = words
            f.write(json.dumps(page, ensure_ascii=False) + "\n")
            n_pages += 1
            n_sections += len(page["sections"])
            n_words += words
    print(f"страниц: {n_pages}, разделов: {n_sections}, слов: {n_words}, пропущено: {len(skipped)}")


if __name__ == "__main__":
    main()
