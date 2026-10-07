"""Настройки проекта: .env, сеть, выбранная конфигурация поиска.

Сеть — главный источник проблем на Windows с VPN, поэтому здесь:
- запросы к Сберу (GigaChat) идут НАПРЯМУЮ, мимо системного прокси: через зарубежный IP Сбер
  может не пустить, а прокси со схемой https:// ломает Python-клиенты (как было с pip);
- модели HuggingFace берутся из локального кэша (скачаны на шаге 2), без обращения в интернет.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --- выбор по итогам шага 2 (results/ablation_dev.md): bge-m3, нарезка по разделам, без гибрида/реранкера
RETRIEVER = {"type": "dense", "model": "BAAI/bge-m3", "batch_size": 8}
CHUNKING = "structural"
CONTEXT_BUDGET_WORDS = 1500   # Recall@1500 слов = 0.94 на dev
MAX_CONTEXT_CHUNKS = 8
SEARCH_K = 20

DEFAULT_CA_FILE = ROOT / "certs" / "russian_trusted_root_ca.crt"
CA_URL = "https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt"
# авторизация — ngw.devices.sberbank.ru, запросы — api.giga.chat (новый адрес в SDK 0.2) или старый домен
SBER_HOSTS = ["sberbank.ru", ".sberbank.ru", "ngw.devices.sberbank.ru", "gigachat.devices.sberbank.ru",
              "giga.chat", ".giga.chat", "api.giga.chat"]
COMBINED_CA_FILE = ROOT / "certs" / "ca_bundle_combined.pem"
MODEL_PREFERENCE = ["GigaChat-3-Ultra", "GigaChat-2-Max", "GigaChat-2-Pro", "GigaChat-2"]

_loaded = False


def load_env() -> None:
    """Читает .env (если есть) и настраивает сеть. Повторные вызовы безопасны."""
    global _loaded
    if _loaded:
        return
    env_path = ROOT / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path, override=False)
        except ImportError:  # без python-dotenv — простой разбор KEY=VALUE
            for line in env_path.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    # сертификат: SDK использует файл как ЕДИНСТВЕННЫЙ источник доверия, поэтому собираем общий файл
    # «стандартные корневые (certifi) + НУЦ Минцифры» — тогда работают и домены Сбера, и api.giga.chat
    ca = os.environ.get("GIGACHAT_CA_BUNDLE_FILE")
    if not ca or Path(ca).resolve() == DEFAULT_CA_FILE.resolve():
        combined = build_combined_ca()
        if combined:
            os.environ["GIGACHAT_CA_BUNDLE_FILE"] = str(combined)
    # Сбер — мимо прокси
    for var in ("NO_PROXY", "no_proxy"):
        cur = [h for h in os.environ.get(var, "").split(",") if h]
        os.environ[var] = ",".join(dict.fromkeys(cur + SBER_HOSTS))
    # модели — из локального кэша
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    _loaded = True


def ensure_ca(timeout: float = 20) -> bool:
    """Скачивает корневой сертификат НУЦ Минцифры, если его нет (нужно и локально, и в Docker)."""
    if DEFAULT_CA_FILE.exists():
        return True
    import urllib.request
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # российский адрес — напрямую
        data = opener.open(CA_URL, timeout=timeout).read()
        if b"BEGIN CERTIFICATE" not in data:
            return False
        DEFAULT_CA_FILE.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_CA_FILE.write_bytes(data)
        return True
    except Exception:  # noqa: BLE001 — без сети просто не получится, check_env.py подскажет
        return False


def build_combined_ca() -> Path | None:
    if not DEFAULT_CA_FILE.exists() and not ensure_ca():
        return None
    try:
        import certifi
        ru = DEFAULT_CA_FILE.read_text(encoding="ascii", errors="ignore")
        if (not COMBINED_CA_FILE.exists()
                or COMBINED_CA_FILE.stat().st_mtime < DEFAULT_CA_FILE.stat().st_mtime):
            COMBINED_CA_FILE.write_text(Path(certifi.where()).read_text(encoding="ascii", errors="ignore")
                                        + "\n" + ru, encoding="ascii")
        return COMBINED_CA_FILE
    except ImportError:
        return DEFAULT_CA_FILE


def mask(value: str | None) -> str:
    if not value:
        return "— (пусто)"
    return f"{value[:4]}…{value[-4:]} ({len(value)} символов)" if len(value) > 12 else "*** (слишком короткий?)"
