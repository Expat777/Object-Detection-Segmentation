"""
Веб-слой: FastAPI-приложение для детекции и анонимизации лиц.

Эндпоинты
---------
GET  /                 -> HTML-страница с формой загрузки
POST /detect           -> обработка из формы, возвращает HTML с результатом
POST /api/detect       -> чистый API: вернёт PNG с размытыми лицами
POST /api/detect/json  -> чистый API: вернёт JSON с координатами лиц
GET  /about            -> метрики и графики обучения модели
GET  /health           -> healthcheck для Docker/мониторинга
"""

import base64
import logging
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import detector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(BASE_DIR, "runs", "detect", "train-28")


def _warmup_model():
    """Грузит модель в фоновом потоке, не блокируя старт сервера."""
    try:
        detector.get_model()
        logger.info("Модель прогрета и готова.")
    except Exception as exc:
        logger.warning("Не удалось прогреть модель: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Старт сервера НЕ должен блокироваться загрузкой модели.

    Если веса уже скачаны — грузим модель в фоновом потоке (страницы при этом
    открываются сразу). Если весов ещё нет, не качаем их на старте: модель
    загрузится лениво при первом запросе к /detect.
    """
    if detector.weights_exist():
        threading.Thread(target=_warmup_model, daemon=True).start()
    else:
        logger.info("Весов нет локально — модель загрузится при первом запросе.")
    yield


app = FastAPI(title="Face Detection & Blur API", version="1.0.0", lifespan=lifespan)

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Графики обучения лежат в runs/ (в .gitignore). Если папка есть — отдаём как статику.
if os.path.isdir(RUNS_DIR):
    app.mount("/runs", StaticFiles(directory=os.path.join(BASE_DIR, "runs")), name="runs")


def _to_data_uri(data: bytes, mime: str = "image/png") -> str:
    """Кодирует байты картинки в data-URI, чтобы вставить прямо в <img src>."""
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


# ============================================================
#  HTML-ФРОНТ
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/detect", response_class=HTMLResponse)
async def detect_form(
    request: Request,
    conf: float = Form(0.15),
    file: UploadFile | None = File(None),
    url: str = Form(""),
):
    # 1. Получаем байты картинки: либо из файла, либо по URL
    try:
        if file is not None and file.filename:
            image_bytes = await file.read()
            source_name = file.filename
            mime = file.content_type or "image/jpeg"
        elif url.strip():
            image_bytes = detector.fetch_image_from_url(url.strip())
            source_name = url.strip().split("/")[-1] or "url_image"
            mime = "image/jpeg"
        else:
            return templates.TemplateResponse(
                request, "index.html",
                {"error": "Загрузите файл или вставьте URL."},
            )
    except Exception as exc:
        return templates.TemplateResponse(
            request, "index.html",
            {"error": f"Не удалось получить изображение: {exc}"},
        )

    # 2. Прогоняем через модель
    try:
        result_png, faces = detector.process_image(image_bytes, conf=conf)
    except Exception as exc:
        return templates.TemplateResponse(
            request, "index.html",
            {"error": f"Ошибка обработки: {exc}"},
        )

    # 3. Рендерим страницу результата (картинки встроены как data-URI)
    return templates.TemplateResponse(
        request, "result.html",
        {
            "source_name": source_name,
            "faces": faces,
            "count": len(faces),
            "conf": conf,
            "original": _to_data_uri(image_bytes, mime),
            "result": _to_data_uri(result_png, "image/png"),
        },
    )


# ============================================================
#  ЧИСТЫЙ REST API (виден в Swagger /docs)
# ============================================================
@app.post("/api/detect")
async def api_detect_image(file: UploadFile = File(...), conf: float = Form(0.15)):
    """Принимает изображение, возвращает PNG с размытыми лицами."""
    image_bytes = await file.read()
    result_png, _ = detector.process_image(image_bytes, conf=conf)
    return Response(content=result_png, media_type="image/png")


@app.post("/api/detect/json")
async def api_detect_json(file: UploadFile = File(...), conf: float = Form(0.15)):
    """Принимает изображение, возвращает JSON с координатами и уверенностью лиц."""
    image_bytes = await file.read()
    _, faces = detector.process_image(image_bytes, conf=conf, draw_boxes=False)
    return JSONResponse({"count": len(faces), "faces": faces})


# ============================================================
#  СТРАНИЦА О МОДЕЛИ
# ============================================================
@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    graphs = {
        "Матрица ошибок (Confusion Matrix)": "confusion_matrix.png",
        "PR-кривая (Precision-Recall)": "BoxPR_curve.png",
        "F1-кривая (F1-Confidence)": "BoxF1_curve.png",
        "Результаты по эпохам": "results.png",
    }
    available = {
        title: f"/runs/detect/train-28/{fname}"
        for title, fname in graphs.items()
        if os.path.exists(os.path.join(RUNS_DIR, fname))
    }
    return templates.TemplateResponse(
        request, "about.html", {"graphs": available}
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
