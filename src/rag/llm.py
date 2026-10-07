"""Клиенты LLM с единым интерфейсом generate(system, user) -> LLMResult.

GigaChat SDK в версии 0.2 сменил API (client.chat.create + ChatCompletionRequest), но сохранил старый
(client.chat(Chat(...)) -> choices[0].message.content). Пробуем новый, при несовпадении — старый;
какой сработал, пишется в result.api — это видно в check_env.py.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field


@dataclass
class LLMResult:
    text: str
    model: str
    latency_s: float
    api: str = ""
    usage: dict = field(default_factory=dict)


class GigaChatLLM:
    def __init__(self, model: str | None = None, temperature: float = 0.2, max_tokens: int = 800,
                 timeout: float = 60.0, max_retries: int = 4):
        from rag.settings import load_env
        load_env()
        from gigachat import GigaChat
        self.model = model or os.environ.get("GIGACHAT_MODEL") or "GigaChat-3-Ultra"
        self.temperature, self.max_tokens, self.max_retries = temperature, max_tokens, max_retries
        kwargs = dict(credentials=os.environ.get("GIGACHAT_CREDENTIALS"),
                      scope=os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
                      model=self.model, timeout=timeout)
        ca = os.environ.get("GIGACHAT_CA_BUNDLE_FILE")
        if ca:
            kwargs["ca_bundle_file"] = ca
        self.client = GigaChat(**kwargs)
        self._api = None  # запоминаем рабочий вариант API после первого успеха

    # --- два варианта вызова SDK ---
    def _call_new(self, system: str, user: str):
        """SDK 0.2: ChatCompletionRequest; температура и лимит — внутри model_options."""
        from gigachat.models import ChatCompletionRequest, ChatMessage
        try:
            from gigachat.models import ChatModelOptions
            opts = ChatModelOptions(temperature=self.temperature, max_tokens=self.max_tokens)
        except ImportError:
            opts = {"temperature": self.temperature, "max_tokens": self.max_tokens}
        req = ChatCompletionRequest(model=self.model, model_options=opts,
                                    messages=[ChatMessage(role="system", content=system),
                                              ChatMessage(role="user", content=user)])
        resp = self.client.chat.create(req)
        content = resp.messages[-1].content
        text = content if isinstance(content, str) else "".join(
            getattr(part, "text", None) or "" for part in (content or []))
        return text, _usage(resp)

    def _call_legacy(self, system: str, user: str):
        from gigachat.models import Chat, Messages, MessagesRole
        payload = Chat(messages=[Messages(role=MessagesRole.SYSTEM, content=system),
                                 Messages(role=MessagesRole.USER, content=user)],
                       model=self.model, temperature=self.temperature, max_tokens=self.max_tokens)
        resp = self.client.chat(payload)
        return resp.choices[0].message.content, _usage(resp)

    def _try_apis(self, system: str, user: str) -> LLMResult:
        """Пробует варианты API; ошибка «не та форма запроса/ответа» — повод попробовать следующий."""
        from gigachat.exceptions import GigaChatException
        errors = []
        for api in ([self._api] if self._api else ["new", "legacy"]):
            t0 = time.time()
            try:
                text, usage = (self._call_new if api == "new" else self._call_legacy)(system, user)
            except GigaChatException:
                raise  # авторизация, лимиты, сеть — это не про форму API
            except (AttributeError, TypeError, ImportError, IndexError, KeyError, ValueError) as e:
                errors.append(f"{api}: {type(e).__name__}: {e}")
                continue
            self._api = api
            return LLMResult(text=(text or "").strip(), model=self.model, latency_s=time.time() - t0,
                             api=api, usage=usage)
        raise RuntimeError("Ни один вариант API GigaChat SDK не подошёл: " + " | ".join(errors))

    def generate(self, system: str, user: str) -> LLMResult:
        """Повторяет запрос при 429 (у бесплатного тарифа — один запрос одновременно)."""
        for attempt in range(self.max_retries):
            try:
                return self._try_apis(system, user)
            except Exception as e:  # noqa: BLE001
                if not _is_rate_limit(e) or attempt == self.max_retries - 1:
                    raise
                time.sleep(float(getattr(e, "retry_after", None) or 2 ** (attempt + 1)))
        raise RuntimeError("unreachable")

    def list_models(self) -> list[str]:
        models = self.client.get_models()
        return [getattr(m, "id_", None) or getattr(m, "id", None) for m in models.data]


def _is_rate_limit(e: Exception) -> bool:
    try:
        from gigachat.exceptions import RateLimitError
        if isinstance(e, RateLimitError):
            return True
    except ImportError:
        pass
    return "429" in str(e)


def _usage(resp) -> dict:
    u = getattr(resp, "usage", None)
    if u is None:
        return {}
    keys = ("prompt_tokens", "completion_tokens", "input_tokens", "output_tokens", "total_tokens")
    return {k: getattr(u, k) for k in keys if getattr(u, k, None) is not None}


class FakeLLM:
    """Для тестов и отладки без сети: возвращает заранее заданный ответ."""

    def __init__(self, reply: str = "Ответ [1]."):
        self.reply, self.model, self.calls = reply, "fake", []

    def generate(self, system: str, user: str) -> LLMResult:
        self.calls.append((system, user))
        text = self.reply(user) if callable(self.reply) else self.reply
        return LLMResult(text=text, model="fake", latency_s=0.0, api="fake")
