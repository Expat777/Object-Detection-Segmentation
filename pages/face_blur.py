import os
import io
import requests
import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# --- НАСТРОЙКА ПУТЕЙ И УСТРОЙСТВА ---
CURRENT_DIR = os.getcwd()

# Локальный путь, куда сохранятся веса внутри проекта
WEIGHTS_PATH = os.path.join(CURRENT_DIR, "models", "face", "best.pt")
# Прямая ссылка на веса в вашем GitHub Releases
WEIGHTS_URL = "https://github.com/Expat777/Object-Detection-Segmentation/releases/download/v01/best.pt"

# Папка с графиками обучения (по умолчанию ищет в проекте, при необходимости измените)
YOLO_RUN_DIR = os.path.join(CURRENT_DIR, "runs", "detect", "train-28")

# Автоматический выбор: GPU (0), если доступен, иначе CPU
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# --- ФУНКЦИЯ ЗАГРУЗКИ МОДЕЛИ ---
@st.cache_resource
def load_models():
    # 1. Создаем папку для весов, если её еще нет
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    
    # 2. Если файла весов нет — скачиваем его с GitHub
    if not os.path.exists(WEIGHTS_PATH):
        with st.spinner("📥 Веса модели не найдены локально. Скачиваем с GitHub Releases... Пожалуйста, подождите."):
            try:
                response = requests.get(WEIGHTS_URL, stream=True, timeout=30)
                response.raise_for_status()  # Вызовет ошибку, если ссылка невалидна
                
                with open(WEIGHTS_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                st.success("✅ Веса модели успешно скачаны!")
            except Exception as e:
                st.error(f"❌ Не удалось автоматически скачать веса: {e}")
                st.info("Проверьте подключение к интернету или доступность ссылки на GitHub.")
                return None

    # 3. Инициализация модели YOLO
    if os.path.exists(WEIGHTS_PATH):
        return YOLO(WEIGHTS_PATH)
    else:
        st.error(f"Файл весов не найден по пути: {WEIGHTS_PATH}")
        return None

# Запускаем загрузку модели
detect_model = load_models()

# Если модель не удалось загрузить — останавливаем выполнение страницы
if detect_model is None:
    st.warning("Работа страницы приостановлена, так как модель не загружена.")
    st.stop()

# --- ЛОКАЛЬНАЯ НАВИГАЦИЯ НА СТРАНИЦЕ ---
page = st.sidebar.selectbox('Выберите подраздел:', ['Детекция и Маскировка лиц', 'О модели и процессе обучения'])

# ==========================================
# ПОДРАЗДЕЛ 1: ДЕТЕКЦИЯ И МАСКИРОВКА
# ==========================================
if page == 'Детекция и Маскировка лиц':
    st.title('👤 Детекция и анонимизация лиц')
    st.write('Загрузите изображения с вашего компьютера или вставьте URL-ссылку на картинку.')
    
    # Формы загрузки данных
    uploaded_files = st.file_uploader(
        'Выберите изображения (одно или несколько):',
        type=['jpeg', 'jpg', 'png'],
        accept_multiple_files=True
    )
    url_input = st.text_input('Или вставьте URL-ссылку на изображение:')

    images_to_process = []

    # Собираем файлы из загрузчика
    if uploaded_files:
        for file in uploaded_files:
            images_to_process.append((file.name, Image.open(file)))

    # Собираем файлы по ссылке
    if url_input:
        try:
            response = requests.get(url_input, timeout=10)
            if response.status_code == 200:
                img_name = url_input.split('/')[-1] or 'url_image.jpg'
                if "?" in img_name:  # Очистка имени от GET-параметров ссылки
                    img_name = img_name.split('?')[0]
                images_to_process.append((img_name, Image.open(io.BytesIO(response.content))))
            else:
                st.error('Не удалось скачать изображение по указанму URL.')
        except Exception as e:
            st.error(f'Ошибка при обработке URL: {e}')

    # Обработка изображений
    if images_to_process:
        st.success(f'🎰 Изображений к обработке: {len(images_to_process)}')

        # Ползунок порога уверенности
        conf_threshold = st.slider('Порог уверенности модели (Confidence)', 0.10, 1.00, 0.15, step=0.05)
        
        for name, img in images_to_process:
            st.markdown(f"---")
            st.subheader(f'📁 Файл: {name}')

            # Конвертация PIL Image -> OpenCV BGR
            img_np = np.array(img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            anon_img = img_bgr.copy()

            # Инференс модели YOLO
            results = detect_model.predict(source=img_bgr, conf=conf_threshold, imgsz=800, device=DEVICE, verbose=False)[0]
            
            if len(results.boxes) > 0:
                for box in results.boxes:
                    # Безопасное извлечение координат рамки
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Вырезаем область лица (ROI)
                    face_roi = anon_img[y1:y2, x1:x2]

                    if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                        # Рассчитываем размер ядра размытия (должно быть нечетным)
                        ksize_x = int(face_roi.shape[1] * 0.4) | 1
                        ksize_y = int(face_roi.shape[0] * 0.4) | 1
                        
                        # Применяем размытие по Гауссу и возвращаем на картинку
                        blur_face = cv2.GaussianBlur(face_roi, (ksize_x, ksize_y), 0)
                        anon_img[y1:y2, x1:x2] = blur_face

                # Рисуем bounding boxes поверх размытого изображения
                annot_img = results.plot(labels=True, conf=True, img=anon_img)
                final_img = cv2.cvtColor(annot_img, cv2.COLOR_BGR2RGB)
            else:
                final_img = img_np

            # Вывод результатов в две колонки
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption='Оригинал', use_container_width=True)
            with col2:
                st.image(final_img, caption='Результат (Размытие лиц)', use_container_width=True)

# ==========================================
# ПОДРАЗДЕЛ 2: ИНФОРМАЦИЯ ОБ ОБУЧЕНИИ
# ==========================================
elif page == 'О модели и процессе обучения':
    st.title('📊 Информация о модели и качестве обучения')
    st.write('Ниже представлены основные метрики и графики процесса обучения нейросети.')
       
    # Карточки с метриками
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label='Эпох обучения (Epochs)', value=25)
    with col2:
        st.metric(label='Размер выборки (Train)', value='13 386 изображений')
    with col3:
        st.metric(label='Точность mAP50', value='0.883')
    with col4:
        st.metric(label='Архитектура модели', value='YOLOv11x (Extra Large)')
        
    st.write('---')
    st.header('📈 Графики обучения')

    # Словарь соответствия графиков
    metrics_mapping = {
        "Матрица ошибок (Confusion Matrix)": "confusion_matrix.png",
        "PR-кривая (Precision-Recall Curve)": "BoxPR_curve.png",
        "F1-кривая (F1-Confidence Curve)": "BoxF1_curve.png",
        "Результаты по эпохам (Loss & Metrics)": "results.png"
    }

    col_g1, col_g2 = st.columns(2)

    # Отображение графиков в сетке из двух колонок
    for idx, (title, filename) in enumerate(metrics_mapping.items()):
        full_path = os.path.join(YOLO_RUN_DIR, filename)
        target_col = col_g1 if idx % 2 == 0 else col_g2
        
        with target_col:
            st.subheader(title)
            if os.path.exists(full_path):
                st.image(full_path, use_container_width=True)
            else:
                st.info(f"График '{filename}' не найден. Для отображения графиков перенесите их в папку: `{YOLO_RUN_DIR}`")
