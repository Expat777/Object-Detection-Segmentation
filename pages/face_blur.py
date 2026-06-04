import os
import io
import requests
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


st.set_page_config(page_title='Face Analys', layout='wide')

@st.cache_resource
def load_models():
    detect_model = YOLO('/home/vitaliy/runs/detect/train-28/weights/best.pt')
    return detect_model

detect_model = load_models()

page = st.sidebar.selectbox('Please choose pages', ['Detect and Blur Face', 'About model and process of fit'])

if page == 'Detect and Blur Face':
    st.title('Detection and Anonymazing Faces')
    st.write('Load pic`s from your PC or use URL on pic`s')
    uploaded_files = st.file_uploader(
        'please choose pic for fit(one or more)',
        type=['jpeg', 'jpg', 'png'],
        accept_multiple_files=True
    )
    url_input = st.text_input('put Here you URL')

    images_to_process = []

    if uploaded_files:
        for file in uploaded_files:
            images_to_process.append((file.name, Image.open(file)))

    if url_input:
        try:
            response = requests.get(url_input, timeout=10)
            if response.status_code == 200:
                img_name = url_input.split('/')[-1] or 'url_image.jpg'
                images_to_process.append((img_name, Image.open(io.BytesIO(response.content))))
            else:
                st.error(f' cannot download files by URL')
        except Exception as e:
            st.error(f'URL problem {e}')

    if images_to_process:
        st.success(f' Ready to fit {len(images_to_process)}')

        conf_threshold = st.slider('Confidence', 0.1, 1.0, 0.15)
        for name, img in images_to_process:
            st.subheader(f' File {name}')

            img_np = np.array(img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            anon_img = img_bgr.copy()

            results =  detect_model.predict(source=img_bgr, conf=conf_threshold, imgsz=800, device=0, verbose=False)[0]
            if len(results.boxes) > 0:
                for box in results.boxes:
                    x1,y1,x2,y2 = map(int, box.xyxy.squeeze().tolist())
                    face_roi = anon_img[y1:y2,x1:x2]

                    if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                        ksize_x = int(face_roi.shape[1] * 0.4) | 1
                        ksize_y = int(face_roi.shape[0] * 0.4) | 1
                        blur_face = cv2.GaussianBlur(face_roi, (ksize_x, ksize_y), 0)
                        anon_img[y1:y2, x1:x2] = blur_face

               
                annot_img = results.plot(labels=True, conf=True, img=anon_img)
                final_img = cv2.cvtColor(annot_img, cv2.COLOR_BGR2RGB)
            else:
                final_img = img_np

            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption='Original')
            with col2:
                st.image(final_img, caption='Blur_Face')

elif page == 'About model and process of fit':
    st.title('Info about model and quality of fit')
       
    yolo_run_dir = '/home/vitaliy/runs/detect/train-28'

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label='Epochs', value=25)
    with col2:
        st.metric(label='Value of train ', value = '13 386 Pic`s')
    with col3:
        st.metric(label='mAP50', value='0.883')
    with col4:
        st.metric(label='model', value ='yolo11x')
    st.write('---')

    st.header('Graphics  fit quality')

    metrics_mapping = {
        "Матрица ошибок (Confusion Matrix)": "confusion_matrix.png",
        "PR-кривая (Precision-Recall Curve)": "BoxPR_curve.png",
        "F1-кривая (F1-Confidence Curve)": "BoxF1_curve.png",
        "Результаты по эпохам (Loss & Metrics)": "results.png"
    }

    col_g1, col_g2 = st.columns(2)

    for idx, (title, filename) in enumerate(metrics_mapping.items()):
        full_path = os.path.join(yolo_run_dir, filename)
        target_col = col_g1 if idx % 2 == 0  else col_g2
        with target_col:
            st.subheader(title)
            if os.path.exists(full_path):
                st.image(full_path)
            else:
                st.info(f'Graph {filename} not found {yolo_run_dir}')
