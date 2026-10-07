"""HTTP API поверх пайплайна.

    uvicorn rag.api:app --app-dir src --port 8000
    POST /ask {"question": "..."}  →  {answer, refused, sources[], latency}
    GET  /health
Документация с кнопкой «Try it out»: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

_state: dict = {}
_lock = threading.Lock()  # бесплатный GigaChat — один запрос одновременно; GPU тоже не любит параллель


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, examples=["Как посмотреть список топиков?"])


class SourceOut(BaseModel):
    n: int
    section: str
    url: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    sources: list[SourceOut]
    retrieval_s: float
    generation_s: float
    model: str


def get_pipeline():
    return _state["rag"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    from rag.llm import GigaChatLLM
    from rag.pipeline import RAGPipeline
    t = time.time()
    _state["rag"] = RAGPipeline(GigaChatLLM())  # модель и индекс грузятся один раз при старте
    _state["ready_s"] = round(time.time() - t, 1)
    yield
    _state.clear()


app = FastAPI(title="ROS 2 RAG assistant", version="0.3", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok" if "rag" in _state else "loading", "startup_s": _state.get("ready_s")}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):  # sync-функция: FastAPI выполнит её в пуле потоков, не блокируя сервер
    try:
        with _lock:
            ans = get_pipeline().answer(req.question.strip())
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {str(e)[:300]}") from e
    return AskResponse(answer=ans.answer, refused=ans.refused,
                       sources=[SourceOut(n=s.n, section=s.section, url=s.url) for s in ans.sources],
                       retrieval_s=ans.retrieval_s, generation_s=ans.generation_s, model=ans.model)
