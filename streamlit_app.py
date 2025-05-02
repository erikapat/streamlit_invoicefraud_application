# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import openai
import io

# Leer la clave de OpenAI desde secrets.toml
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(
    page_title="Resúmenes por Documento",
    page_icon="🔍",
    layout="wide"
)

st.title("Resúmenes por Documento")

# Diccionario de traducción ampliado para analysis_name y mensajes comunes
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
    # Añade más traducciones según tus necesidades
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

No incluyas una lista de problemas detectados, solo los dos apartados anteriores.

Validaciones fallidas ({failed_count} de {total_validations}):
"""
    for _, row in failed_validations.iterrows():
        validation_type = translate(row['analysis_name']) if pd.notna(row['analysis_name']) else "Validación desconocida"
        message = translate(row['comments']) if pd.notna(row['comments']) else "No hay detalles adicionales"
        prompt += f"\n• {validation_type}: {message}"

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "Eres un asistente especializado en detección de fraude en facturas que ayuda a interpretar resultados de validaciones automáticas."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error al generar el resumen: {str(e)}"

uploaded_file = st.file_uploader("Carga un archivo CSV con resultados de validación", type=["csv"])

if uploaded_file is not None:
    try:
        content = uploaded_file.read()
        content_str = content.decode('utf-8')
        delimiter = ';' if ';' in content_str.split('\n')[0] else ','
        df = pd.read_csv(io.StringIO(content_str), delimiter=delimiter)

        required_columns = ['doc_name', 'is_valid', 'comments', 'analysis_name']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(f"Faltan columnas requeridas en el archivo: {', '.join(missing_columns)}")
        else:
            def to_bool(x):
                if pd.isna(x):
                    return None
                if x is True or str(x).lower() == 'true':
                    return True
                if x is False or str(x).lower() == 'false':
                    return False
                return None

            df['is_valid'] = df['is_valid'].apply(to_bool)

            # Ordena facturas por número de validaciones fallidas
            fail_counts = df[df['is_valid'] == False].groupby('doc_name').size().reset_index(name='fail_count')
            all_docs = pd.DataFrame({'doc_name': df['doc_name'].unique()})
            fail_counts = all_docs.merge(fail_counts, on='doc_name', how='left').fillna(0)
            fail_counts['fail_count'] = fail_counts['fail_count'].astype(int)
            sorted_doc_names = fail_counts.sort_values(['fail_count', 'doc_name'], ascending=[False, True])['doc_name'].tolist()

            selected_doc = st.selectbox(
                "Selecciona la factura (doc_name) que quieres analizar (ordenadas por nº de validaciones fallidas):",
                options=sorted_doc_names
            )

            document_data = df[df['doc_name'] == selected_doc]

            # --- Estructura en columnas ---
            col1, col2 = st.columns([1.2, 2])

            with col1:
                st.subheader("Estadísticas de Validación")
                valid_count = document_data['is_valid'].value_counts()
                successful = valid_count.get(True, 0)
                failed = valid_count.get(False, 0)
                na_count = document_data['is_valid'].isna().sum()

                st.markdown(f"<span style='color:green'>✅ Validaciones exitosas: {successful}</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='color:red'>❌ Validaciones fallidas: {failed}</span>", unsafe_allow_html=True)
                if na_count > 0:
                    st.markdown(f"<span style='color:orange'>⚠️ Validaciones sin resultado: {na_count}</span>", unsafe_allow_html=True)

                if failed > 0:
                    st.subheader("Validaciones Fallidas")
                    failed_validations = document_data[document_data['is_valid'] == False]
                    for _, row in failed_validations.iterrows():
                        validation_type = translate(row['analysis_name']) if pd.notna(row['analysis_name']) else "Validación desconocida"
                        message = translate(row['comments']) if pd.notna(row['comments']) else "Sin detalles"
                        st.markdown(f"<b>{validation_type}:</b> {message}", unsafe_allow_html=True)

            with col2:
                st.subheader("Resumen de Anomalías")
                with st.spinner(f"Generando resumen para {selected_doc}..."):
                    summary = generate_summary(selected_doc, document_data)
                    st.markdown(summary)

    except Exception as e:
        st.error(f"Error al procesar el archivo: {str(e)}")
        st.info("Asegúrate de que el archivo CSV tenga el formato correcto y contenga las columnas necesarias.")

else:
    st.info("Carga un archivo CSV para comenzar el análisis")
    with st.expander("Formato esperado del archivo CSV"):
        st.markdown("""
        El archivo debe contener las siguientes columnas:

        - **doc_name**: Identificador del documento o factura
        - **is_valid**: Resultado de la validación (true/false)
        - **comments**: Mensaje o descripción cuando la validación falla
        - **analysis_name**: Tipo de validación realizada

        Cada fila representa una validación individual aplicada a un documento.
        Las validaciones con `is_valid = false` indican anomalías que requieren revisión.
        """)
