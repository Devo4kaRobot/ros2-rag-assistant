"""Ответ пайплайна → HTML-сообщение Telegram (без внешних зависимостей, тестируется отдельно)."""
from __future__ import annotations

import html
import re

TG_LIMIT = 4096


def md_to_tg_html(text: str) -> str:
    """Экранируем HTML, затем ```блоки``` → <pre>, `код` → <code>, **жирный** → <b>."""
    out, pos = [], 0
    for m in re.finditer(r"```[\w+-]*\n?(.*?)```", text, re.S):
        out.append(_inline(text[pos:m.start()]))
        out.append(f"<pre>{html.escape(m.group(1).strip())}</pre>")
        pos = m.end()
    out.append(_inline(text[pos:]))
    return "".join(out)


def _inline(s: str) -> str:
    parts = re.split(r"(`[^`\n]+`)", s)
    res = []
    for p in parts:
        if p.startswith("`") and p.endswith("`") and len(p) > 1:
            res.append(f"<code>{html.escape(p[1:-1])}</code>")
        else:
            e = html.escape(p)
            e = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", e)
            res.append(e)
    return "".join(res)


def render_answer(answer: str, sources: list[dict], refused: bool) -> str:
    body = md_to_tg_html(answer)
    if refused or not sources:
        tail = "" if refused else "\n\n<i>Ответ без ссылок на документацию — относись к нему осторожно.</i>"
        return _fit(body + tail)
    by_url: dict[str, tuple[str, list[int]]] = {}
    for s in sources:
        by_url.setdefault(s["url"], (s["section"], []))[1].append(s["n"])
    lines = [f'[{", ".join(map(str, ns))}] <a href="{html.escape(url, quote=True)}">{html.escape(sec)}</a>'
             for url, (sec, ns) in by_url.items()]
    return _fit(body + "\n\n<b>Источники:</b>\n" + "\n".join(lines))


def _fit(s: str) -> str:
    if len(s) <= TG_LIMIT:
        return s
    # режем по границе строки, чтобы не порвать HTML-тег посередине
    cut = s.rfind("\n", 0, TG_LIMIT - 40)
    return s[: cut if cut > 0 else TG_LIMIT - 40] + "\n…(ответ обрезан)"
