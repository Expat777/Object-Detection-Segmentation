# Face Detection & Blur — FastAPI

Сервис для детекции и анонимизации (размытия) лиц на изображениях.
Модель — YOLOv11x, обученная на 13 386 изображениях (mAP50 = 0.883).

## Возможности

| Эндпоинт | Метод | Что делает |
|---|---|---|
| `/` | GET | HTML-форма загрузки изображения |
| `/detect` | POST | Обработка из формы → HTML с результатом |
| `/api/detect` | POST | Принять картинку → вернуть PNG с размытыми лицами |
| `/api/detect/json` | POST | Принять картинку → вернуть JSON с координатами лиц |
| `/about` | GET | Метрики и графики обучения |
| `/docs` | GET | Swagger UI (автодокументация API) |
| `/health` | GET | Healthcheck |

Веса модели (`models/face/best.pt`) скачиваются автоматически при первом
запуске с [GitHub Releases](https://github.com/Expat777/Object-Detection-Segmentation/releases),
поэтому в репозиторий и Docker-образ их класть не нужно.

## Локальный запуск

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Открыть http://localhost:8000

## Запуск в Docker

```bash
docker compose up --build
```

или вручную:

```bash
docker build -t face-blur .
docker run -p 8000:8000 -v face-weights:/app/models face-blur
```

> Volume `models` сохраняет скачанные веса между перезапусками контейнера.

## Деплой на VPS

```bash
git clone <repo> && cd Object-Detection-Segmentation
docker compose up -d --build
```

За приложением рекомендуется поставить reverse-proxy (nginx / Caddy) для HTTPS.

## Структура

```
app.py             # FastAPI: эндпоинты и веб-слой
detector.py        # ML-логика: загрузка модели + размытие лиц
templates/         # HTML-шаблоны (Jinja2)
requirements.txt
Dockerfile
docker-compose.yml
```
