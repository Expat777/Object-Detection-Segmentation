FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Системные библиотеки для opencv-python-headless / ultralytics
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libxcb1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-сборки torch И torchvision ставим вместе с одного индекса —
# иначе torchvision притянется PyPI-сборкой под CUDA и упадёт
# с "operator torchvision::nms does not exist".
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY . .

# Веса (~гигабайты) скачиваются при первом старте с GitHub Releases,
# поэтому в образ их не кладём (см. .dockerignore).

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
