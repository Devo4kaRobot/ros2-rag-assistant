"""Лёгкий парсер reStructuredText документации ROS 2 → чистый текст по разделам.

Sphinx не нужен: обрабатываем только те конструкции, что реально встречаются в
ros2_documentation (заголовки, директивы, вкладки, роли). Решения:
- вкладки ОС: оставляем Linux/Ubuntu, Windows/macOS/RHEL отбрасываем (иначе
  одна инструкция трижды попадает в индекс);
- остальные вкладки (C++/Python, XML/YAML, CMake/Python) сохраняем с подписью;
- код из code-block сохраняем целиком в ``` блоках;
- literalinclude подставляем из файла, если он есть и не длиннее 120 строк.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

DISTRO = "lyrical"
SOURCE_ROOT: list[Path] = []  # заполняется в parse_page, нужен для include с абсолютным путём
SUBST_VARS = {
    "{DISTRO}": DISTRO,
    "{DISTRO_TITLE}": DISTRO.title(),
    "{DISTRO_TITLE_FULL}": "Lyrical Luth",
    "{DISTRO_UBUNTU_DEB_PLATFORM}": "Ubuntu Resolute Raccoon (26.04)",
    "{REPOS_FILE_BRANCH}": DISTRO,
    "{DISTRO_ARM_STATUS_SUFFIX}": "unv8",
}
DROP_TABS = {"windows", "macos", "rhel"}
SKIP_DIRECTIVES = {
    "redirect-from", "contents", "toctree", "image", "figure", "meta", "raw", "ifconfig", "graphviz", "mermaid", "adopters-table", "only",
    "highlight", "index", "spelling:word-list", "youtube",
}
CODE_DIRECTIVES = {"code-block", "code", "sourcecode"}
ADMONITIONS = {"note", "warning", "tip", "attention", "caution", "important", "hint", "danger", "seealso"}
UNDERLINE_RE = re.compile(r"^([=\-^~\"*#+`':.])\1{2,}\s*$")
DIRECTIVE_RE = re.compile(r"^(\s*)\.\.\s+([\w:-]+)::\s*(.*)$")


def make_anchor(title: str) -> str:
    """Как docutils.make_id: латиница в нижнем регистре, дефисы, без ведущих цифр."""
    s = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    s = re.sub(r"^[^a-z]+", "", s)
    return s


def load_substitutions(path: Path) -> dict[str, str]:
    subs = {}
    if path.exists():
        for m in re.finditer(r"^\.\. \|([^|]+)\| replace:: (.+)$", path.read_text(encoding="utf-8"), re.M):
            subs[m.group(1)] = m.group(2)
    return subs


def clean_inline(text: str, subs: dict[str, str]) -> str:
    for k, v in SUBST_VARS.items():
        text = text.replace(k, v)
    text = re.sub(r"\|([^|\s][^|]*?)\|_{0,2}", lambda m: subs.get(m.group(1), m.group(0)), text)
    # :role:`текст <цель>` → текст ; :role:`цель` → цель (для :doc: — последний сегмент пути)
    def role(m):
        name, body = m.group(1), m.group(2)
        mm = re.match(r"(.*?)\s*<([^>]*)>$", body, re.S)
        if mm and mm.group(1):
            return mm.group(1)
        target = mm.group(2) if mm else body
        if name == "doc":
            target = target.rstrip("/").split("/")[-1].replace("-", " ")
        return target.lstrip("~!")
    text = re.sub(r":([\w:+-]+):`([^`]+)`", role, text)
    text = re.sub(r"`([^`<]+?)\s*<[^>]+>`__?", r"\1", text)   # внешние ссылки
    text = re.sub(r"`([^`]+)`_\b", r"\1", text)
    text = text.replace("``", "`")
    return text


@dataclass
class Section:
    titles: list[str]          # путь заголовков от корня страницы
    anchor: str
    lines: list[str] = field(default_factory=list)
    html_anchor: str = ""

    @property
    def text(self) -> str:
        t = "\n".join(self.lines)
        return re.sub(r"\n{3,}", "\n\n", t).strip()


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _block_end(lines: list[str], start: int, base_indent: int) -> int:
    """Индекс первой строки после тела директивы (тело — строки с отступом > base_indent или пустые)."""
    i = start
    while i < len(lines):
        ln = lines[i]
        if ln.strip() and _indent(ln) <= base_indent:
            break
        i += 1
    return i


def _dedent(block: list[str]) -> list[str]:
    ind = min((_indent(l) for l in block if l.strip()), default=0)
    return [l[ind:] if l.strip() else "" for l in block]


def render_block(lines: list[str], src_dir: Path, subs: dict[str, str]) -> list[str]:
    """Рекурсивно превращает фрагмент rst (без заголовков) в чистые строки."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = DIRECTIVE_RE.match(ln)
        if m:
            ind, name, arg = len(m.group(1)), m.group(2).lower(), m.group(3).strip()
            end = _block_end(lines, i + 1, ind)
            body = lines[i + 1:end]
            # опции директивы (:language: и т. п.) идут первыми строками тела
            opts, k = {}, 0
            while k < len(body) and re.match(r"^\s+:[\w-]+:", body[k]):
                om = re.match(r"^\s+:([\w-]+):\s*(.*)$", body[k])
                opts[om.group(1)] = om.group(2)
                k += 1
            content = _dedent(body[k:])
            if name in SKIP_DIRECTIVES:
                pass
            elif name in CODE_DIRECTIVES:
                lang = arg or opts.get("language", "")
                code = "\n".join(content).strip("\n")
                for kk, vv in SUBST_VARS.items():
                    code = code.replace(kk, vv)
                out += ["", f"```{lang}", code, "```", ""]
            elif name == "include":
                # подключаемые .rst-фрагменты (обычно _Имя.rst) вставляем как обычный текст
                p = (src_dir / arg.lstrip("/")).resolve() if not arg.startswith("/") else None
                if arg.startswith("/"):
                    p = (SOURCE_ROOT[0] / arg.lstrip("/")).resolve() if SOURCE_ROOT else None
                if p is not None and p.exists() and p.suffix in (".rst", ".txt"):
                    inc = p.read_text(encoding="utf-8", errors="ignore").replace("\t", "    ").splitlines()
                    inc = [l for l in inc if not UNDERLINE_RE.match(l.strip())]
                    out += render_block(inc, p.parent, subs)
            elif name == "literalinclude":
                p = (src_dir / arg).resolve()
                if p.exists() and p.is_file():
                    txt = p.read_text(encoding="utf-8", errors="ignore").splitlines()
                    if len(txt) <= 120:
                        out += ["", f"```{opts.get('language', '')}", *txt, "```", ""]
                    else:
                        out += [f"[код из файла {arg}, {len(txt)} строк]"]
                else:
                    out += [f"[код из файла {arg}]"]
            elif name == "tabs":
                out += render_block(content, src_dir, subs)
            elif name in ("group-tab", "tab"):
                label = clean_inline(arg, subs)
                if label.lower() in DROP_TABS:
                    pass
                elif label.lower() in ("linux", "ubuntu"):
                    out += render_block(content, src_dir, subs)
                else:
                    out += ["", f"**{label}:**"] + render_block(content, src_dir, subs)
            elif name in ADMONITIONS:
                inner = render_block(content, src_dir, subs)
                if arg:
                    inner = [clean_inline(arg, subs)] + inner
                out += ["", f"{name.capitalize()}: " + (inner[0] if inner else "")] + inner[1:]
            elif name == "list-table":
                if arg:
                    out.append(clean_inline(arg, subs))
                for row in "\n".join(content).split("\n* "):
                    cells = [clean_inline(c.strip().replace("\n", " "), subs)
                             for c in re.split(r"\n?\s*- ", row.lstrip("* ")) if c.strip()]
                    if cells:
                        out.append(" | ".join(cells))
            else:  # glossary, math, versionadded, неизвестное — оставляем содержимое
                if arg and name not in ("glossary",):
                    out.append(clean_inline(arg, subs))
                out += render_block(content, src_dir, subs)
            i = end
            continue
        stripped = ln.strip()
        if stripped.startswith(".. _") or stripped == ".." or re.match(r"^\.\. \|", stripped) \
                or (stripped.startswith("..") and not stripped.startswith("...") and "::" not in stripped):
            # метки, комментарии, определения подстановок — пропускаем вместе с телом
            end = _block_end(lines, i + 1, _indent(ln))
            i = end if stripped in ("..",) or stripped.startswith(".. |") else i + 1
            continue
        if stripped.endswith("::") and not stripped.startswith(".."):
            # литеральный блок: "Пример::" + блок с отступом
            end = _block_end(lines, i + 1, _indent(ln))
            text = stripped[:-2].rstrip()
            if text:
                out.append(clean_inline(text + ":", subs))
            code = "\n".join(_dedent(lines[i + 1:end])).strip("\n")
            for kk, vv in SUBST_VARS.items():
                code = code.replace(kk, vv)
            if code:
                out += ["", "```", code, "```", ""]
            i = end
            continue
        if UNDERLINE_RE.match(stripped) and stripped[0] == "+":
            i += 1  # рамки grid-таблиц
            continue
        out.append(clean_inline(ln.rstrip(), subs) if stripped else "")
        i += 1
    return out


def parse_page(path: Path, source_root: Path, subs: dict[str, str]) -> dict:
    raw = path.read_text(encoding="utf-8", errors="ignore").replace("\t", "    ").splitlines()
    page_id = path.relative_to(source_root).with_suffix("").as_posix()
    # 1) находим заголовки: строка текста + подчёркивание (с необязательной надчёркивающей линией)
    heads = []  # (line_idx_start, line_idx_after, title, style)
    i = 0
    while i < len(raw) - 1:
        a, b = raw[i].rstrip(), raw[i + 1].rstrip()
        if UNDERLINE_RE.match(raw[i].strip()) and i + 2 < len(raw) and UNDERLINE_RE.match(raw[i + 2].strip()) \
                and raw[i + 1].strip() and raw[i][0] == raw[i + 2][0] and _indent(raw[i]) == 0:
            heads.append((i, i + 3, raw[i + 1].strip(), "over" + raw[i][0]))
            i += 3
            continue
        if a.strip() and not a.startswith(" ") and UNDERLINE_RE.match(b) and _indent(raw[i + 1]) == 0 \
                and len(b) >= len(a.strip()) - 2 and b[0] != "+" and not DIRECTIVE_RE.match(a):
            heads.append((i, i + 2, a.strip(), b[0]))
            i += 2
            continue
        i += 1
    styles: list[str] = []
    for h in heads:
        if h[3] not in styles:
            styles.append(h[3])
    src_dir = path.parent
    if not SOURCE_ROOT:
        SOURCE_ROOT.append(source_root)
    # 2) режем страницу по заголовкам
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    preamble_end = heads[0][0] if heads else len(raw)
    pre = render_block(raw[:preamble_end], src_dir, subs)
    title = clean_inline(heads[0][2], subs) if heads else page_id.split("/")[-1]
    if any(l.strip() for l in pre):
        sections.append(Section([title], "", pre))
    for n, (s, e, t, st) in enumerate(heads):
        level = styles.index(st)
        t_clean = clean_inline(t, subs)
        stack = [x for x in stack if x[0] < level] + [(level, t_clean)]
        nxt = heads[n + 1][0] if n + 1 < len(heads) else len(raw)
        body = render_block(raw[e:nxt], src_dir, subs)
        sections.append(Section([x[1] for x in stack], make_anchor(t_clean), body))
    # якорь — ключ раздела, уникальный внутри страницы (повторы получают суффикс ~2, ~3);
    # html_anchor — то, что ставится в ссылку на docs.ros.org
    seen: dict[str, int] = {}
    for sec in sections:
        a = sec.anchor
        seen[a] = seen.get(a, 0) + 1
        sec.html_anchor = a
        if seen[a] > 1:
            sec.anchor = f"{a}~{seen[a]}"
    url = f"https://docs.ros.org/en/{DISTRO}/{page_id}.html"
    return {
        "page_id": page_id,
        "title": title,
        "url": url,
        "sections": [
            {"titles": s.titles, "anchor": s.anchor, "html_anchor": s.html_anchor, "text": s.text}
            for s in sections if s.text or len(s.titles) > 0
        ],
    }
