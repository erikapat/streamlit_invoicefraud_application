# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import openai
import io
import os
import json
import subprocess
from pathlib import Path

# Leer la clave de OpenAI desde secrets.toml
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(
    page_title="Análisis de Facturas",
    page_icon="🔍",
    layout="wide"
)

st.title("Análisis de Facturas")

# Diccionario de traducción
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


def generate_summary(doc_name, validation_data):
    failed_validations = validation_data[validation_data['is_valid'] == False]
    total_validations = len(validation_data)
    failed_count = len(failed_validations)

    if failed_count == 0:
        return "✅ Todas las validaciones realizadas han sido exitosas. No se han detectado problemas que requieran revisión manual."

    prompt = f"""
Eres un experto en detección de fraude en facturas. Analiza los siguientes resultados de validación 
para la factura '{doc_name}' y genera un resumen conciso en español que ayude al usuario a entender:

- Elementos para revisar manualmente (lista breve y concreta)
- Posibles riesgos o indicios de fraude

Validaciones fallidas ({failed_count} de {total_validations}):
"""
    for _, row in failed_validations.iterrows():
        validation_type = translate(row['analysis_name']) if pd.notna(
            row['analysis_name']) else "Validación desconocida"
        message = translate(row['comments']) if pd.notna(row['comments']) else "No hay detalles adicionales"
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
        return f"Error al generar resumen: {str(e)}"


def find_file_with_extension(folder_path, filename_without_ext):
    folder = Path(folder_path)
    for file_path in folder.rglob(f"{filename_without_ext}.*"):
        return str(file_path)
    return None


# Selector de carpeta usando subprocess (solo local)
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


# Interfaz principal
uploaded_file = st.file_uploader("Carga un archivo CSV con resultados de validación", type=["csv"])

# Selección de carpeta
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

        required_columns = ['doc_name', 'is_valid', 'comments', 'analysis_name']
        if missing := [col for col in required_columns if col not in df.columns]:
            st.error(f"Faltan columnas: {', '.join(missing)}")
            st.stop()

        # Procesamiento de datos
        df['is_valid'] = df['is_valid'].apply(lambda x: x if pd.isna(x) else str(x).lower() == 'true')

        fail_counts = (df[df['is_valid'] == False]
                       .groupby('doc_name').size()
                       .reset_index(name='fail_count'))
        all_docs = pd.DataFrame({'doc_name': df['doc_name'].unique()})
        fail_counts = all_docs.merge(fail_counts, on='doc_name', how='left').fillna(0)
        max_fails = fail_counts['fail_count'].max() or 1
        fail_counts['risk_score'] = (fail_counts['fail_count'] / max_fails * 98).round(1)

        sorted_docs = fail_counts.sort_values(['fail_count', 'doc_name'],
                                              ascending=[False, True])['doc_name'].tolist()

        selected_doc = st.selectbox(
            "Selecciona la factura a analizar:",
            options=sorted_docs,
            format_func=lambda
                x: f"{x} (Riesgo: {fail_counts.loc[fail_counts['doc_name'] == x, 'risk_score'].iloc[0]}%)"
        )

        doc_data = df[df['doc_name'] == selected_doc]
        risk_score = fail_counts.loc[fail_counts['doc_name'] == selected_doc, 'risk_score'].iloc[0]
        failed_validations = doc_data[doc_data['is_valid'] == False]

        # Layout en columnas
        col1, col2 = st.columns([1.2, 2])

        with col1:
            st.markdown(f"## 🔴 Nivel de riesgo: **{risk_score}%**")
            st.progress(risk_score / 100)

            st.markdown("### 📊 Estadísticas")
            valid_count = doc_data['is_valid'].value_counts()
            successful = valid_count.get(True, 0)
            failed = valid_count.get(False, 0)
            na_count = doc_data['is_valid'].isna().sum()

            st.markdown(f"<span style='color:green'>✅ Validaciones exitosas: {successful}</span>",
                        unsafe_allow_html=True)
            st.markdown(f"<span style='color:red'>❌ Validaciones fallidas: {failed}</span>", unsafe_allow_html=True)
            if na_count > 0:
                st.markdown(f"<span style='color:orange'>⚠️ Validaciones sin resultado: {na_count}</span>",
                            unsafe_allow_html=True)

        with col2:
            st.markdown("## 📄 Resumen de anomalías")
            with st.spinner("Generando resumen con IA..."):
                summary = generate_summary(selected_doc, doc_data)
                st.markdown(summary)

        # Validaciones fallidas
        st.divider()
        st.markdown("## ❌ Validaciones Fallidas")
        #st.markdown(f"<span style='color:red'>❌ Validaciones fallidas: {failed}</span>", unsafe_allow_html=True)
        if not failed_validations.empty:
            for _, row in failed_validations.iterrows():
                st.markdown(f"**{translate(row['analysis_name'])}**: {translate(row['comments'])}")
        else:
            st.markdown("No se encontraron validaciones fallidas")

        # Mostrar factura relacionada
        if folder_path:
            st.divider()
            st.markdown("## 📑 Factura asociada")
            invoice_file = find_file_with_extension(folder_path, selected_doc)
            if invoice_file:
                ext = os.path.splitext(invoice_file)[1].lower()
                if ext in [".jpg", ".jpeg", ".png"]:
                    st.image(invoice_file, use_container_width=True)
                elif ext == ".pdf":
                    with open(invoice_file, "rb") as f:
                        st.download_button("Descargar PDF", f, file_name=os.path.basename(invoice_file))
                else:
                    st.markdown(f"Archivo encontrado: `{invoice_file}` (formato no soportado)")
            else:
                st.warning("No se encontró la factura en la carpeta seleccionada")

    except Exception as e:
        st.error(f"Error crítico: {str(e)}")
        st.stop()
else:
    st.info("ℹ️ Carga un archivo CSV para comenzar el análisis")
