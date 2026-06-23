"""
ML-слой приложения: загрузка YOLO-модели и обработка изображений.

Здесь нет ничего, связанного с вебом (FastAPI). Этот модуль можно
импортировать откуда угодно и тестировать отдельно от сервера.
"""

import io
import os
import tempfile
import threading

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from ultralytics import YOLO

# --- ПУТИ И УСТРОЙСТВО ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Локальный путь, куда сохранятся веса внутри проекта
WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "face", "best.pt")
# Прямая ссылка на веса в GitHub Releases
WEIGHTS_URL = "https://github.com/Expat777/Object-Detection-Segmentation/releases/download/v01/best.pt"

# Авто-выбор: GPU (0), если доступен, иначе CPU
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Модель грузим один раз и держим в памяти (ленивая инициализация).
_model: YOLO | None = None
# Блокировка: чтобы при параллельных запросах не качать/грузить модель дважды.
_lock = threading.Lock()


def weights_exist() -> bool:
    """True, если веса уже скачаны локально."""
    return os.path.exists(WEIGHTS_PATH)


def _download_weights() -> None:
    """Скачивает веса с GitHub Releases, если их ещё нет локально.

    Загрузка атомарная: качаем во временный файл с уникальным именем и
    переименовываем в итоговый только после успешного завершения. Так
    оборванная или параллельная закачка не оставит битый файл на месте весов.
    """
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    if weights_exist():
        return

    # Уникальный временный файл в той же папке (чтобы os.replace был атомарным
    # и не конфликтовал с другими закачками).
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(WEIGHTS_PATH), suffix=".part")
    os.close(fd)
    try:
        # timeout=(connect, read): 10 c на соединение, 300 c на чтение чанка
        with requests.get(WEIGHTS_URL, stream=True, timeout=(10, 300)) as response:
            response.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        os.replace(tmp_path, WEIGHTS_PATH)  # атомарное переименование
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)  # подчищаем недокачанный файл
        raise


def get_model() -> YOLO:
    """Возвращает singleton-экземпляр модели, при необходимости загрузив веса."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:  # повторная проверка под блокировкой
                _download_weights()
                _model = YOLO(WEIGHTS_PATH)
    return _model


def _bytes_to_bgr(data: bytes) -> np.ndarray:
    """Декодирует байты картинки (PNG/JPG) в OpenCV-массив формата BGR."""
    pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    rgb = np.array(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def process_image(
    data: bytes,
    conf: float = 0.15,
    draw_boxes: bool = True,
) -> tuple[bytes, list[dict]]:
    """
    Находит лица, размывает их по Гауссу и (опц.) рисует рамки.

    Параметры
    ---------
    data : bytes
        Сырые байты входного изображения.
    conf : float
        Порог уверенности модели.
    draw_boxes : bool
        Рисовать ли bounding box'ы поверх размытого изображения.

    Возвращает
    ----------
    (png_bytes, faces)
        png_bytes — обработанное изображение, закодированное в PNG;
        faces — список словарей {"box": [x1,y1,x2,y2], "conf": float}.
    """
    model = get_model()
    img_bgr = _bytes_to_bgr(data)

    results = model.predict(
        source=img_bgr,
        conf=conf,
        imgsz=800,
        device=DEVICE,
        verbose=False,
    )[0]

    anon = img_bgr.copy()
    faces: list[dict] = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = round(float(box.conf[0]), 3)
        faces.append({"box": [x1, y1, x2, y2], "conf": confidence})

        roi = anon[y1:y2, x1:x2]
        if roi.shape[0] > 0 and roi.shape[1] > 0:
            # Ядро размытия должно быть нечётным -> | 1
            ksize_x = int(roi.shape[1] * 0.4) | 1
            ksize_y = int(roi.shape[0] * 0.4) | 1
            anon[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (ksize_x, ksize_y), 0)

    if draw_boxes and len(results.boxes) > 0:
        # results.plot возвращает аннотированный массив в формате BGR
        anon = results.plot(labels=True, conf=True, img=anon)

    ok, buffer = cv2.imencode(".png", anon)
    if not ok:
        raise RuntimeError("Не удалось закодировать результат в PNG")

    return buffer.tobytes(), faces


def fetch_image_from_url(url: str) -> bytes:
    """Скачивает изображение по URL и возвращает его байты."""
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return response.content
