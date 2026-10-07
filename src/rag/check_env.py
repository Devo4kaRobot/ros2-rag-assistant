"""Проверка окружения перед запуском бота. Ключи НЕ печатаются (только маска).

    python src/rag/check_env.py

1) .env и ключи  2) сертификат НУЦ Минцифры (скачивается сам, если нет)  3) GigaChat: модели и тестовый
запрос  4) Telegram: токен бота  5) модель поиска в локальном кэше
"""
from __future__ import annotations

import json
import os
import ssl
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rag.settings import CA_URL, DEFAULT_CA_FILE, MODEL_PREFERENCE, load_env, mask  # noqa: E402

OK, BAD, WARN = "[OK] ", "[!!] ", "[??] "
problems: list[str] = []


def direct_opener():
    """urllib без прокси — для российских адресов."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def check_env_file():
    print("\n1) Файл .env")
    if not (ROOT / ".env").exists():
        print(BAD + "нет файла .env — скопируй .env.example в .env и заполни"); problems.append(".env")
        return
    load_env()
    for k in ("GIGACHAT_CREDENTIALS", "GIGACHAT_SCOPE", "TELEGRAM_BOT_TOKEN"):
        v = os.environ.get(k, "")
        print((OK if v else BAD) + f"{k} = {mask(v) if 'SCOPE' not in k else v or '—'}")
        if not v:
            problems.append(k)
    cred = os.environ.get("GIGACHAT_CREDENTIALS", "")
    if cred and (" " in cred or cred.startswith(("'", '"'))):
        print(WARN + "в ключе GigaChat есть пробел или кавычка — проверь, что вставлен ровно ключ")


def check_cert():
    print("\n2) Сертификат НУЦ Минцифры (нужен для HTTPS к GigaChat)")
    path = Path(os.environ.get("GIGACHAT_CA_BUNDLE_FILE") or DEFAULT_CA_FILE)
    if not path.exists():
        print("   файла нет, скачиваю с Госуслуг…")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = direct_opener().open(CA_URL, timeout=30).read()
            if b"BEGIN CERTIFICATE" not in data:
                raise ValueError("скачалось что-то не то (нет BEGIN CERTIFICATE)")
            path.write_bytes(data)
            os.environ["GIGACHAT_CA_BUNDLE_FILE"] = str(path)
        except Exception as e:  # noqa: BLE001
            print(BAD + f"не скачался: {e}\n      Скачай вручную: {CA_URL}\n      и положи как {DEFAULT_CA_FILE}")
            problems.append("сертификат")
            return
    try:
        ssl.create_default_context(cafile=str(path))
        print(OK + f"{path}")
    except Exception as e:  # noqa: BLE001
        print(BAD + f"файл есть, но не читается как сертификат: {e}"); problems.append("сертификат")


def check_gigachat():
    print("\n3) GigaChat")
    if not os.environ.get("GIGACHAT_CREDENTIALS"):
        print(BAD + "пропускаю: нет ключа"); return
    try:
        from importlib.metadata import version
        print(OK + f"библиотека gigachat {version('gigachat')}")
        from rag.llm import GigaChatLLM
        llm = GigaChatLLM(model=os.environ.get("GIGACHAT_MODEL") or "GigaChat-2")
        models = [m for m in llm.list_models() if m]
        print(OK + "доступные модели: " + ", ".join(models))
        chosen = os.environ.get("GIGACHAT_MODEL")
        if not chosen:
            chosen = next((m for m in MODEL_PREFERENCE if m in models), models[0] if models else "GigaChat-2")
            print(WARN + f"GIGACHAT_MODEL не задан — беру {chosen}. Добавь в .env строку: GIGACHAT_MODEL={chosen}")
        llm.model = chosen
        r = llm.generate("Отвечай одним словом.", "Столица России?")
        print(OK + f"модель {chosen} ответила «{r.text[:60]}» за {r.latency_s:.1f} с (API SDK: {r.api}, токены: {r.usage})")
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        print(BAD + msg[:500])
        if "401" in msg or "Authentication" in msg:
            print("      → ключ не принят: перевыпусти Authorization key на developers.sber.ru и вставь заново")
        elif "CERTIFICATE" in msg.upper() or "SSL" in msg.upper():
            print("      → проблема с сертификатом: проверь пункт 2")
        elif "402" in msg:
            print("      → закончилась квота у этой модели: задай другую в GIGACHAT_MODEL")
        problems.append("GigaChat")


def check_telegram():
    print("\n4) Telegram-бот")
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print(BAD + "пропускаю: нет токена"); return
    url = f"https://api.telegram.org/bot{token}/getMe"
    for label, opener in (("напрямую", direct_opener()), ("через прокси из окружения", urllib.request.build_opener())):
        try:
            data = json.loads(opener.open(url, timeout=15).read())
            if data.get("ok"):
                print(OK + f"бот @{data['result']['username']} отвечает ({label})")
                if label != "напрямую":
                    print(WARN + "Telegram доступен только через прокси — бот возьмёт его из HTTPS_PROXY")
                return
        except urllib.error.HTTPError as e:
            if e.code == 401:
                print(BAD + "токен не принят Telegram (401) — проверь, что скопирован целиком"); problems.append("Telegram")
                return
        except Exception as e:  # noqa: BLE001
            print(WARN + f"{label}: {type(e).__name__}: {str(e)[:120]}")
    problems.append("Telegram")


def check_index():
    print("\n5) Модель поиска (bge-m3) в локальном кэше")
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache("BAAI/bge-m3", "config.json")
        print((OK if isinstance(hit, str) else BAD) + ("есть" if isinstance(hit, str) else "нет — запусти шаг 2 с интернетом"))
        idx = list((ROOT / "data" / "index").glob("BAAI__bge-m3__fp32__*.npy"))
        print((OK if idx else WARN) + (f"эмбеддинги корпуса посчитаны ({len(idx)} файл.)" if idx
                                      else "эмбеддингов в fp32 ещё нет — посчитаются при первом запуске"))
    except Exception as e:  # noqa: BLE001
        print(WARN + f"не удалось проверить: {e}")


if __name__ == "__main__":
    t = time.time()
    check_env_file(); check_cert(); check_gigachat(); check_telegram(); check_index()
    print("\n" + ("Всё готово ✔" if not problems else "Нужно поправить: " + ", ".join(dict.fromkeys(problems)))
          + f"  ({time.time() - t:.0f} с)")
