"""Telegram-бот.

    python -m bot.main          (из папки src)   или   python src/bot/main.py

По умолчанию пайплайн работает в том же процессе (одна копия модели в памяти — важно для 8 ГБ ОЗУ).
Если задан API_URL (например http://127.0.0.1:8000), бот ходит в HTTP API — так будет в Docker.
Если Telegram доступен только через прокси, бот возьмёт его из TELEGRAM_PROXY или HTTPS_PROXY.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aiogram import Bot, Dispatcher, F  # noqa: E402
from aiogram.client.default import DefaultBotProperties  # noqa: E402
from aiogram.enums import ParseMode  # noqa: E402
from aiogram.filters import Command, CommandStart  # noqa: E402
from aiogram.types import (CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup,  # noqa: E402
                           LinkPreviewOptions, Message)
from aiogram.utils.chat_action import ChatActionSender  # noqa: E402

from bot.formatting import render_answer  # noqa: E402
from rag.settings import load_env  # noqa: E402

log = logging.getLogger("bot")
FEEDBACK = ROOT / "results" / "feedback.jsonl"
EXAMPLES = ["Как посмотреть, какие топики сейчас есть?",
            "Чем сервис отличается от action?",
            "Как развести ROS 2 в одной Wi-Fi сети с коллегами, чтобы не видеть чужие топики?"]

dp = Dispatcher()
_answerer = None
_lock = asyncio.Lock()


class LocalAnswerer:
    def __init__(self):
        from rag.llm import GigaChatLLM
        from rag.pipeline import RAGPipeline
        self.rag = RAGPipeline(GigaChatLLM())

    async def ask(self, q: str) -> dict:
        ans = await asyncio.to_thread(self.rag.answer, q)  # поиск и HTTP к GigaChat — в отдельном потоке
        return {"answer": ans.answer, "refused": ans.refused,
                "sources": [{"n": s.n, "section": s.section, "url": s.url} for s in ans.sources],
                "generation_s": ans.generation_s, "retrieval_s": ans.retrieval_s}


class ApiAnswerer:
    def __init__(self, url: str):
        self.url = url.rstrip("/") + "/ask"

    async def ask(self, q: str) -> dict:
        import aiohttp
        async with aiohttp.ClientSession() as s:
            async with s.post(self.url, json={"question": q}, timeout=aiohttp.ClientTimeout(total=120)) as r:
                r.raise_for_status()
                return await r.json()


def feedback_kb(qid: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="👍", callback_data=f"fb:{qid}:1"),
        InlineKeyboardButton(text="👎", callback_data=f"fb:{qid}:0")]])


def log_event(rec: dict):
    FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


@dp.message(CommandStart())
async def start(m: Message):
    ex = "\n".join(f"• {e}" for e in EXAMPLES)
    await m.answer("Привет! Я отвечаю на вопросы по <b>ROS 2</b> (Lyrical Luth) по официальной документации "
                   "и даю ссылки на разделы docs.ros.org. Если в документации ответа нет — так и скажу.\n\n"
                   f"Например:\n{ex}")


@dp.message(Command("help"))
async def help_(m: Message):
    await m.answer("Просто напиши вопрос обычным текстом. Под ответом есть 👍/👎 — это помогает улучшать бота.")


@dp.message(F.text)
async def question(m: Message):
    q = m.text.strip()
    if len(q) < 3:
        await m.answer("Сформулируй вопрос чуть подробнее 🙂")
        return
    qid = uuid.uuid4().hex[:10]
    t = time.time()
    try:
        # «печатает…» держится, пока идёт ответ (обычный send_chat_action гаснет через 5 с)
        async with ChatActionSender.typing(bot=m.bot, chat_id=m.chat.id):
            async with _lock:  # по одному вопросу за раз: лимит бесплатного GigaChat и одна видеокарта
                res = await _answerer.ask(q)
    except Exception as e:  # noqa: BLE001
        log.exception("ошибка ответа")
        await m.answer("Не получилось ответить — попробуй ещё раз через минуту.")
        log_event({"qid": qid, "question": q, "error": f"{type(e).__name__}: {e}", "ts": time.time()})
        return
    await m.answer(render_answer(res["answer"], res["sources"], res["refused"]),
                   reply_markup=feedback_kb(qid), link_preview_options=LinkPreviewOptions(is_disabled=True))
    log_event({"qid": qid, "user": m.from_user.id if m.from_user else None, "question": q,
               "answer": res["answer"], "refused": res["refused"], "sources": [s["url"] for s in res["sources"]],
               "total_s": round(time.time() - t, 2), "ts": time.time()})


@dp.callback_query(F.data.startswith("fb:"))
async def feedback(c: CallbackQuery):
    _, qid, val = c.data.split(":")
    log_event({"qid": qid, "feedback": int(val), "ts": time.time()})
    await c.answer("Спасибо!")
    if c.message:
        await c.message.edit_reply_markup(reply_markup=None)


async def main():
    global _answerer
    load_env()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        sys.exit("Нет TELEGRAM_BOT_TOKEN в .env")
    api_url = os.environ.get("API_URL")
    log.info("Загружаю пайплайн (%s)…", "через API " + api_url if api_url else "локально")
    _answerer = ApiAnswerer(api_url) if api_url else LocalAnswerer()
    session = None
    # Telegram у части пользователей открывается только через VPN-прокси: берём TELEGRAM_PROXY,
    # а если его нет — обычный HTTPS_PROXY из окружения (GigaChat при этом всё равно идёт напрямую)
    proxy = (os.environ.get("TELEGRAM_PROXY") or os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
             or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy"))
    if proxy:
        from aiogram.client.session.aiohttp import AiohttpSession
        session = AiohttpSession(proxy=proxy)
        log.info("Telegram через прокси %s", proxy)
    bot = Bot(token, session=session, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    me = await bot.get_me()
    log.info("Бот @%s запущен. Остановить — Ctrl+C", me.username)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
