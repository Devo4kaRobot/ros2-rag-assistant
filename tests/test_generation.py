"""Тесты пайплайна ответов на фиктивной LLM и фиктивном поиске (без сети и GPU)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bot.formatting import TG_LIMIT, md_to_tg_html, render_answer  # noqa: E402
from rag.eval_generation import summarize  # noqa: E402
from rag.llm import FakeLLM  # noqa: E402
from rag.pipeline import (REFUSAL, RAGPipeline, build_user_prompt, is_refusal,  # noqa: E402
                          parse_citations, select_by_budget)

CHUNKS = [
    {"chunk_id": f"c{i}", "page_id": f"P{i}", "title": f"Page {i}", "section": f"Page {i} > Sec",
     "url": f"https://docs.ros.org/en/lyrical/P{i}.html#sec", "anchors": ["sec"], "text": "word " * 400}
    for i in range(5)
]


class FakeRetriever:
    def search(self, q, k=10):
        return list(range(len(CHUNKS)))[:k]


def make(reply):
    return RAGPipeline(FakeLLM(reply), retriever=FakeRetriever(), chunks=CHUNKS)


def test_budget_selection():
    sel = select_by_budget(CHUNKS, 1500, 8)
    assert len(sel) == 3                      # 3 × 400 = 1200 ≤ 1500, четвёртый не влезает
    assert len(select_by_budget([{"text": "w " * 5000}], 1500, 8)) == 1   # первый берётся всегда
    assert len(select_by_budget(CHUNKS, 10_000, 2)) == 2


def test_parse_citations():
    assert parse_citations("a [1] b [3][1] c [2, 7]", 3) == ([1, 3, 2], [7])
    assert parse_citations("без ссылок", 3) == ([], [])


def test_refusal_detection():
    assert is_refusal(REFUSAL)
    assert is_refusal("В документации ROS 2 я этого не нашел.")      # «е» вместо «ё»
    assert not is_refusal("Используйте `ros2 topic list` [1].")


def test_pipeline_answer_with_sources():
    ans = make("Выполните `ros2 topic list` [2]. Это показывает топики [1][9].").answer("топики?")
    assert not ans.refused
    assert [s.n for s in ans.sources] == [2, 1]
    assert ans.invalid_citations == [9]
    assert len(ans.context) == 3


def test_pipeline_refusal_has_no_sources():
    ans = make(REFUSAL).answer("как настроить CUDA?")
    assert ans.refused and ans.sources == []


def test_prompt_contains_numbered_fragments_and_question():
    p = build_user_prompt("вопрос?", CHUNKS[:2])
    assert "[1] (страница: Page 0" in p and "[2] (страница: Page 1" in p and p.endswith("Вопрос: вопрос?")


def test_tg_html_escaping_and_code():
    s = md_to_tg_html("Запусти `ros2 run <pkg>` и **готово** <script>\n```bash\necho <1>\n```")
    assert "<code>ros2 run &lt;pkg&gt;</code>" in s
    assert "<b>готово</b>" in s and "&lt;script&gt;" in s
    assert "<pre>echo &lt;1&gt;</pre>" in s


def test_render_answer_groups_sources_and_fits_limit():
    srcs = [{"n": 1, "section": "A", "url": "u1"}, {"n": 3, "section": "A", "url": "u1"},
            {"n": 2, "section": "B & C", "url": "u2"}]
    out = render_answer("ответ", srcs, refused=False)
    assert "[1, 3] <a href=\"u1\">A</a>" in out and "B &amp; C" in out
    assert len(render_answer("x\n" * 5000, srcs, False)) <= TG_LIMIT


def test_summarize_metrics():
    rows = [
        {"answerable": True, "refused": False, "context_hit": True, "citation_hit": True,
         "invalid_citations": [], "cited_urls": ["u"], "generation_s": 2.0, "model": "m"},
        {"answerable": True, "refused": True, "context_hit": True, "citation_hit": False,
         "invalid_citations": [], "cited_urls": [], "generation_s": 1.0, "model": "m"},
        {"answerable": False, "refused": True, "context_hit": None, "citation_hit": None,
         "invalid_citations": [], "cited_urls": [], "generation_s": 1.0, "model": "m"},
    ]
    s = summarize(rows)
    assert s["correct_refusal"] == 1.0 and s["false_refusal"] == 0.5 and s["citation_hit"] == 1.0


def test_refusal_kinds_on_real_answers():
    from rag.pipeline import refusal_kind
    assert refusal_kind(REFUSAL) == "strict"
    assert refusal_kind("Я не нашёл информации о том, как изменить порт roscore в ROS 1.") == "soft"
    assert refusal_kind("В предоставленных фрагментах нет информации о параметрах `urg_node`. [1]") == "soft"
    long_hallucination = ("В документации ROS 2 не указаны версии драйвера. " + "Ставьте nvidia-driver-535 и CUDA. " * 10)
    assert refusal_kind(long_hallucination) == ""          # длинный ответ «от себя» — не отказ
    assert refusal_kind("Остального в документации ROS 2 нет.") == ""  # частичный ответ — не отказ


def test_prompt_versions_selectable():
    from rag.pipeline import PROMPTS
    p = RAGPipeline(FakeLLM("ok [1]"), retriever=FakeRetriever(), chunks=CHUNKS, prompt="v1")
    p.answer("q")
    assert p.llm.calls[-1][0] == PROMPTS["v1"]
    assert set(PROMPTS) >= {"v1", "v2"}
