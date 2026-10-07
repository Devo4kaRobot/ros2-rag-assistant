"""Быстрые тесты: парсер, нарезка, BM25, метрики. Запуск: pytest -q"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.bm25 import BM25, tokenize  # noqa: E402
from rag.chunking import chunk_pages  # noqa: E402
from rag.metrics import is_relevant, mrr_at_k, recall_at_k  # noqa: E402
from rag.rst_parser import clean_inline, make_anchor, parse_page  # noqa: E402

RST = """
Title of page
=============

Intro text.

Section one
-----------

.. tabs::

   .. group-tab:: Linux

      .. code-block:: console

        source /opt/ros/{DISTRO}/setup.bash

   .. group-tab:: Windows

      .. code-block:: console

        call C:\\dev\\setup.bat

See :doc:`the tutorial <../Foo/Bar>` and ``ros2 topic list``.

Section one
-----------

Duplicate title.
"""


def _page(tmp_path):
    src = tmp_path / "source"
    src.mkdir()
    (src / "Page.rst").write_text(RST, encoding="utf-8")
    return parse_page(src / "Page.rst", src, {})


def test_make_anchor():
    assert make_anchor("1 Create a package") == "create-a-package"
    assert make_anchor("What is QoS?") == "what-is-qos"


def test_clean_inline_roles():
    assert clean_inline(":doc:`the tutorial <../Foo/Bar>`", {}) == "the tutorial"
    assert clean_inline(":ref:`target-label`", {}) == "target-label"
    assert clean_inline("``ros2 run``", {}) == "`ros2 run`"


def test_parser_tabs_distro_and_unique_anchors(tmp_path):
    page = _page(tmp_path)
    text = "\n".join(s["text"] for s in page["sections"])
    assert "/opt/ros/lyrical/setup.bash" in text      # {DISTRO} подставлен
    assert "setup.bat" not in text                    # вкладка Windows выброшена
    assert "the tutorial" in text and ":doc:" not in text
    anchors = [s["anchor"] for s in page["sections"]]
    assert len(anchors) == len(set(anchors))          # повторный заголовок получил ~2
    assert "section-one~2" in anchors


def test_chunk_strategies_cover_page(tmp_path):
    page = _page(tmp_path)
    page["n_words"] = 40
    for st in ("fixed", "recursive", "structural"):
        chunks = chunk_pages([page], st)
        assert chunks, st
        assert all(c.page_id == "Page" and c.anchors for c in chunks)


def test_tokenize_splits_underscores():
    toks = tokenize("export ROS_DOMAIN_ID=5")
    assert "ros_domain_id" in toks and "domain" in toks


def test_bm25_ranks_matching_doc_first():
    bm = BM25(["colcon build the workspace", "ros2 topic list shows topics", "unrelated text here"])
    assert bm.search("ros2 topic list", 1)[0][0] == 1


def test_metrics():
    gold = [{"page_id": "P", "anchor": "a"}]
    ranked = [{"page_id": "Q", "anchors": ["a"]}, {"page_id": "P", "anchors": ["b", "a"]}]
    assert is_relevant(ranked[1], gold) and not is_relevant(ranked[0], gold)
    assert recall_at_k(ranked, gold, 1) == 0.0 and recall_at_k(ranked, gold, 2) == 1.0
    assert mrr_at_k(ranked, gold) == 0.5


def test_eval_set_is_consistent():
    """Все gold-ссылки тестового набора существуют в корпусе (если корпус собран)."""
    pages_path = ROOT / "data/processed/pages.jsonl"
    if not pages_path.exists():
        return
    anchors = {(p["page_id"], s["anchor"]) for p in map(json.loads, pages_path.open(encoding="utf-8"))
               for s in p["sections"]}
    qs = [json.loads(l) for l in (ROOT / "eval/questions.jsonl").open(encoding="utf-8") if l.strip()]
    assert len({q["id"] for q in qs}) == len(qs)
    for q in qs:
        assert q["answerable"] == bool(q["gold"]), q["id"]
        for g in q["gold"]:
            assert (g["page_id"], g["anchor"]) in anchors, (q["id"], g)
