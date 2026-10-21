import streamlit as st
import requests

API_URL = 'http://127.0.0.1:8000/pdd/predict/'

st.title('Распознавание дорожных знаков ПДД')

file = st.file_uploader('Загрузите фото знака', type=['png', 'jpg', 'jpeg'])

if file is not None:
    st.image(file, caption='Загруженное фото', width=300)

    if st.button('Определить знак'):
        with st.spinner('Подождите, модель анализирует фото...'):
            try:
                response = requests.post(
                    API_URL,
                    files={'file': (file.name, file.getvalue(), file.type)},
                    timeout=130,
                )
            except requests.exceptions.RequestException:
                st.error('Не удалось подключиться к серверу')
                st.stop()


        if response.status_code == 200:
            data = response.json()
            st.success('Знак определён')
            st.write(f"Номер класса: {data['class_id']}")
            st.write(f"Название: {data['class_name']}")
            st.write(f"Категория: {data['category']}")
            st.write(f"Описание: {data['description']}")
        else:
            try:
                err_message = response.json().get('detail', 'Ошибка')
            except Exception:
                err_message = response.text or f"Ошибка сервера (код {response.status_code})"
            st.error(err_message)