# Образ для API и бота (CPU). Корпус собирается при сборке образа из официальной документации,
# модели и эмбеддинги хранятся в томах, чтобы не скачивать/пересчитывать их при каждом запуске.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models HF_HUB_OFFLINE=0

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-версия torch: образ в разы меньше, а поиск по одному запросу на CPU занимает ~0.1–0.3 с
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY eval/ eval/
COPY configs/ configs/

# корпус: документация ROS 2, ветка lyrical, ЗАКРЕПЛЁННЫЙ коммит (по нему размечен тестовый набор)
ARG DOCS_COMMIT=08601b3c41cf4be349eae5774bc92a9594788641
RUN mkdir -p data/raw/ros2_documentation \
    && git -C data/raw/ros2_documentation init -q \
    && git -C data/raw/ros2_documentation fetch -q --depth 1 https://github.com/ros2/ros2_documentation.git ${DOCS_COMMIT} \
    && git -C data/raw/ros2_documentation checkout -q FETCH_HEAD \
    && python src/rag/build_corpus.py \
    && rm -rf data/raw

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD python -c "import urllib.request,json,sys; sys.exit(0 if json.load(urllib.request.urlopen('http://127.0.0.1:8000/health'))['status']=='ok' else 1)"

CMD ["uvicorn", "rag.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
