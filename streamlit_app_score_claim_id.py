# -*- coding: utf-8 -*-
# streamlit run streamlit_app_test.py
import streamlit as st
import pandas as pd
import openai
import io
import os
import subprocess
from pathlib import Path
from PIL import Image

# Leer la clave de OpenAI desde secrets.toml
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(
    page_title="Global Sentinel - Fraude documental",
    page_icon="🔍",
    layout="wide"
)

st.title("Global Sentinel - Fraude documental")

translation_dict = {
    "Validate Issuer ID number": "Validar número de identificación del emisor",
    "Validate Client ID number": "Validar número de identificación del cliente",
    "Validate provider address in the document country": "Validar dirección del proveedor en el país del documento",
    "Validate client address in the document country": "Validar dirección del cliente en el país del documento",
    "Detect anomalous currency format": "Detectar formato de moneda anómalo",
    "Check multiple languages": "Detectar mezcla de idiomas",
    "NIF number": "El número de identificación fiscal (NIF)",
    "does not match the NIF/NIE patterns": "no coincide con los patrones de NIF/NIE",
    "is not located in the document country": "no se encuentra en el país del documento",
    "The extracted currency": "La moneda extraída",
    "does not match with the document country currency": "no coincide con la moneda del país del documento",
    "Multiple languages detected": "Se detectaron múltiples idiomas",
}

def translate(text):
    if not isinstance(text, str):
        return text
    for eng, esp in translation_dict.items():
        text = text.replace(eng, esp)
    return text

def generate_summary(doc_name, validation_data, doc_type=None):
    failed_validations = validation_data[validation_data['is_valid'] == False]
    total_validations = len(validation_data)
    failed_count = len(failed_validations)

    if failed_count == 0:
        return "✅ Todas las validaciones realizadas han sido exitosas. No se han detectado problemas que requieran revisión manual."

    prompt = f"""
Eres un experto en detección de fraude en facturas. Analiza los siguientes resultados de validación 
para la factura '{doc_name}' de tipo '{doc_type}' y genera un resumen conciso en español que incluya:

- Tipo de documento
- Número total de validaciones
- Validaciones fallidas (con descripción)
- Riesgos o indicios de fraude

Validaciones fallidas ({failed_count} de {total_validations}):
"""

    for _, row in failed_validations.iterrows():
        validation_type = translate(row.get('analysis_name', 'Validación desconocida'))
        message = translate(row.get('comments', 'No hay detalles'))
        prompt += f"\n• {validation_type}: {message}"

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un asistente especializado en detección de fraude en facturas."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error en generación de resumen IA: {e}")
        return "⚠️ No se pudo generar resumen IA."

def find_file_with_extension(folder_path, filename_without_ext):
    folder = Path(folder_path)
    for file_path in folder.rglob(f"{filename_without_ext}.*"):
        return str(file_path)
    return None

def select_folder_via_subprocess():
    try:
        result = subprocess.run(
            ["python", "-c", "import tkinter as tk; from tkinter import filedialog; print(filedialog.askdirectory())"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        st.error(f"Error al seleccionar carpeta: {str(e)}")
    return None

uploaded_file = st.file_uploader("Carga un archivo CSV con resultados de validación", type=["csv"])

folder_path = ""
if st.button("📁 Seleccionar carpeta de facturas"):
    selected_path = select_folder_via_subprocess()
    if selected_path and os.path.isdir(selected_path):
        st.session_state.folder_path = selected_path
        st.success(f"Carpeta seleccionada: {selected_path}")
    else:
        st.warning("No se seleccionó ninguna carpeta válida")

folder_path = st.session_state.get("folder_path", "")

if uploaded_file:
    try:
        content = uploaded_file.read().decode('utf-8')
        delimiter = ';' if ';' in content.split('\n')[0] else ','
        df = pd.read_csv(io.StringIO(content), delimiter=delimiter)

        if 'analysis_score' not in df.columns or 'record_id' not in df.columns:
            st.error("El archivo debe contener las columnas 'analysis_score' y 'record_id'.")
            st.stop()

        df['is_valid'] = df['is_valid'].apply(
            lambda x: True if str(x).lower() == 'true'
            else False if str(x).lower() == 'false'
            else pd.NA
        )

        # Vista por expediente
        st.subheader("📁 Vista por expediente (record_id)")
        expediente_stats = df[df['is_valid'] == False].groupby('record_id').agg(
            avg_score=('analysis_score', 'mean'),
            n_failed=('is_valid', 'count'),
            n_docs=('doc_name', 'nunique')
        ).reset_index()

        selected_record = st.selectbox("Selecciona un expediente (record_id):", expediente_stats.sort_values(by='avg_score', ascending=False)['record_id'])

        df_record = df[df['record_id'] == selected_record]

        st.markdown(f"### 📊 Resumen global del expediente `{selected_record}`")
        avg_score_global = df_record[df_record['is_valid'] == False]['analysis_score'].mean()
        st.markdown(f"**🔴 Score promedio del expediente:** {round(avg_score_global, 1)}%")
        st.markdown(f"**📄 Documentos asociados:** {df_record['doc_name'].nunique()}")
        st.markdown(f"**❌ Validaciones fallidas:** {df_record['is_valid'].value_counts().get(False, 0)}")

        # Selección de documento
        st.subheader("📄 Análisis individual de documento")
        sorted_docs = df_record['doc_name'].dropna().unique()
        selected_doc = st.selectbox("Selecciona un documento:", sorted_docs)

        doc_data = df_record[df_record['doc_name'] == selected_doc]
        risk_score = doc_data[doc_data['is_valid'] == False]['analysis_score'].mean()

        col1, col2 = st.columns([1.2, 2])
        with col1:
            doc_type = doc_data['doc_classification'].iloc[0] if 'doc_classification' in doc_data.columns else "Desconocido"
            st.markdown(f"### 🗂️ Tipo de documento: **{doc_type}**")
            st.markdown(f"## 🔴 Nivel de riesgo: **{round(risk_score, 1)}%**")
            st.progress(min(max(risk_score / 100, 0), 1))

            valid_count = doc_data['is_valid'].value_counts(dropna=False)
            successful = valid_count.get(True, 0)
            failed = valid_count.get(False, 0)
            na_count = doc_data['is_valid'].isna().sum()

            st.markdown("### 📊 Estadísticas")
            st.markdown(f"<span style='color:green'>✅ Validaciones exitosas: {successful}</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='color:red'>❌ Validaciones fallidas: {failed}</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='color:orange'>⚠️ Validaciones sin resultado: {na_count}</span>", unsafe_allow_html=True)

        with col2:
            st.markdown("## 📄 Resumen de anomalías")
            with st.spinner("Generando resumen con IA..."):
                summary = generate_summary(selected_doc, doc_data, doc_type)
                st.markdown(summary)

        st.markdown("## ❌ Validaciones Fallidas")
        for _, row in doc_data[doc_data['is_valid'] == False].iterrows():
            st.markdown(f"**{translate(row['analysis_name'])}**: {translate(row['comments'])}")

    except Exception as e:
        st.error(f"Error crítico: {str(e)}")
else:
    st.info("ℹ️ Carga un archivo CSV para comenzar el análisis")
