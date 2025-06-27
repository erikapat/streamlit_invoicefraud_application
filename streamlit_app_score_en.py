# -*- coding: utf-8 -*-
# To run: streamlit run streamlit_app_test.py

import streamlit as st
import pandas as pd
import openai
import io
import os
import subprocess
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Load OpenAI key from secrets.toml
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Page configuration
st.set_page_config(
    page_title="Global Sentinel - Document Forgery",
    page_icon="🔍",
    layout="wide"
)

st.title("Global Sentinel - Document Forgery")

# Translation dictionary (UI or validation name translations)
translation_dict = {
    "Validate Issuer ID number": "Validate issuer's ID number",
    "Validate Client ID number": "Validate client's ID number",
    "Validate provider address in the document country": "Validate provider's address in the document country",
    "Validate client address in the document country": "Validate client's address in the document country",
    "Detect anomalous currency format": "Detect anomalous currency format",
    "Check multiple languages": "Detect mixed languages",
    "NIF number": "Tax Identification Number (NIF)",
    "does not match the NIF/NIE patterns": "does not match NIF/NIE patterns",
    "is not located in the document country": "is not located in the document country",
    "The extracted currency": "The extracted currency",
    "does not match with the document country currency": "does not match the document country's currency",
    "Multiple languages detected": "Multiple languages detected",
}

# English explanations for known validation failures
explanation_dict = {
    "Validate suspicious software in document Producer":
        "The software used to create the document is unusual or suspicious. It may be a non-standard editing tool.",

    "Validate duplicate invoice in the historical data":
        "An invoice with the same number was found in the historical records. Check if it's a duplicate or an error.",

    "Validate file size with respect to the historical data":
        "The file size is very different from what’s expected based on previous documents. This may indicate tampering.",

    "Validate producer metadata with respect to the historical data":
        "The software used to create the document differs from what is normally used. This may be a sign of editing or artificial generation.",

    "Validate issuer image keypoints against historical data":
        "The issuer image in this document differs from the one that usually appears in correctly validated invoices. It may have been modified or replaced.",

    "Validate number of images in pdf against historical data":
        "The number of images in the PDF does not match what is typical. This could indicate an alteration."
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
        return "✅ All validations passed successfully. No issues requiring manual review were found."

    prompt = f"""
You are an expert in invoice fraud detection. Analyze the following validation results 
for invoice '{doc_name}' and generate a concise summary in English to help the user understand:

- Items that require manual review (brief and clear list)
- Potential risks or fraud indicators

Failed validations ({failed_count} of {total_validations}):
"""
    for _, row in failed_validations.iterrows():
        validation_type = translate(row['analysis_name']) if pd.notna(row['analysis_name']) else "Unknown validation"
        message = explanation_dict.get(validation_type, translate(row['comments']) if pd.notna(row['comments']) else "No additional details")
        prompt += f"\n• {validation_type}: {message}"

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant specialized in invoice fraud detection."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating summary: {str(e)}"

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
        st.error(f"Error selecting folder: {str(e)}")
    return None

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file with validation results", type=["csv"])

folder_path = ""
if st.button("📁 Select invoice folder"):
    selected_path = select_folder_via_subprocess()
    if selected_path and os.path.isdir(selected_path):
        st.session_state.folder_path = selected_path
        st.success(f"Selected folder: {selected_path}")
    else:
        st.warning("No valid folder selected")

folder_path = st.session_state.get("folder_path", "")

if uploaded_file:
    try:
        content = uploaded_file.read().decode('utf-8')
        delimiter = ';' if ';' in content.split('\n')[0] else ','
        df = pd.read_csv(io.StringIO(content), delimiter=delimiter)

        if 'analysis_score' not in df.columns:
            st.error("Missing column: 'analysis_score'")
            st.stop()

        required_columns = ['doc_name', 'is_valid', 'comments', 'analysis_name']
        if missing := [col for col in required_columns if col not in df.columns]:
            st.error(f"Missing columns: {', '.join(missing)}")
            st.stop()

        df['is_valid'] = df['is_valid'].apply(
            lambda x: True if str(x).lower() == 'true'
            else False if str(x).lower() == 'false'
            else pd.NA
        )

        doc_info_all = df.groupby('doc_name').agg({'doc_classification': 'first'}).reset_index()

        df_failed = df[(df['is_valid'] == False) & (df['analysis_score'] > 0)]
        doc_scores = df_failed.groupby('doc_name').agg({'analysis_score': 'mean'}).reset_index().rename(columns={'analysis_score': 'avg_score'})
        fail_counts = df[df['is_valid'] == False].groupby('doc_name').size().reset_index(name='n_failed')

        doc_info = doc_info_all.merge(doc_scores, on='doc_name', how='left').merge(fail_counts, on='doc_name', how='left')
        doc_info['avg_score'] = doc_info['avg_score'].fillna(0)
        doc_info['n_failed'] = doc_info['n_failed'].fillna(0)

        doc_info = doc_info.sort_values(by=['avg_score', 'n_failed'], ascending=[False, False])
        sorted_docs = doc_info['doc_name'].tolist()

        if not sorted_docs:
            st.warning("⚠️ No invoices with failed validations and positive score were found.")
            st.dataframe(df.head())
            st.stop()

        selected_doc = st.selectbox(
            "Select an invoice to analyze:",
            options=sorted_docs,
            format_func=lambda x: f"{x} - {doc_info.loc[doc_info['doc_name'] == x, 'doc_classification'].iloc[0]} (Risk: {round(doc_info.loc[doc_info['doc_name'] == x, 'avg_score'].iloc[0], 1)}%)"
        )

        doc_data = df[df['doc_name'] == selected_doc]
        risk_score = doc_data[doc_data['is_valid'] == False]['analysis_score'].mean()

        col1, col2 = st.columns([1.2, 2])

        with col1:
            classification = doc_data['doc_classification'].iloc[0] if 'doc_classification' in doc_data.columns else "Not available"
            st.markdown(f"### 🗂️ Document type: **{classification}**")

            if risk_score is not None:
                st.markdown(f"## 🔴 Risk level: **{round(risk_score, 1)}%**")
                st.progress(min(max(risk_score / 100, 0), 1))
            else:
                st.markdown("## 🔴 Risk level: Not available")

            st.markdown("### 📊 Validation statistics")
            valid_count = doc_data['is_valid'].value_counts(dropna=False)
            successful = valid_count.get(True, 0)
            failed = valid_count.get(False, 0)
            na_count = doc_data['is_valid'].isna().sum()

            st.markdown(f"<span style='color:green'>✅ Successful: {successful}</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='color:red'>❌ Failed: {failed}</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='color:orange'>⚠️ No result: {na_count}</span>", unsafe_allow_html=True)

        with col2:
            st.markdown("## 📄 AI-generated anomaly summary")
            with st.spinner("Generating summary with AI..."):
                summary = generate_summary(selected_doc, doc_data)
                st.markdown(summary)

        st.divider()
        st.markdown("## ❌ Failed Validations")
        failed_rows = doc_data[doc_data['is_valid'] == False]
        if not failed_rows.empty:
            for _, row in failed_rows.iterrows():
                validation = translate(row['analysis_name'])
                comment = explanation_dict.get(validation, translate(row['comments']))
                st.markdown(f"**{validation}**: {comment}")
        else:
            st.markdown("No failed validations found")

        if folder_path:
            st.divider()
            st.markdown("## 📑 Associated Invoice")
            invoice_file = find_file_with_extension(folder_path, selected_doc)
            if invoice_file:
                ext = os.path.splitext(invoice_file)[1].lower()
                if ext in [".jpg", ".jpeg", ".png", ".jfif"]:
                    img = Image.open(invoice_file)
                    st.image(img, caption=os.path.basename(invoice_file), use_container_width=True)
                elif ext == ".pdf":
                    with open(invoice_file, "rb") as f:
                        st.download_button("Download PDF", f, file_name=os.path.basename(invoice_file))
                else:
                    st.markdown(f"File found: `{invoice_file}` (unsupported format)")
            else:
                st.warning("Invoice file not found in the selected folder")

        st.divider()
        st.markdown("## 📊 Aggregated Validation Failures")

        fail_counts = (df[df['is_valid'] == False]
                       .groupby('doc_name')
                       .size()
                       .reset_index(name='fail_count'))

        all_docs = df[['doc_name']].drop_duplicates()
        fail_counts = all_docs.merge(fail_counts, on='doc_name', how='left').fillna(0)

        total_invoices = df['doc_name'].nunique()
        st.markdown(f"**Total invoices analyzed:** {total_invoices}")
        st.markdown(f"**Invoices with ≥ 1 failure:** {len(fail_counts[fail_counts['fail_count'] > 0])}")
        st.markdown(f"**Invoices with ≥ 2 failures:** {len(fail_counts[fail_counts['fail_count'] >= 2])}")
        st.markdown(f"**Invoices with ≥ 3 failures:** {len(fail_counts[fail_counts['fail_count'] >= 3])}")
        st.markdown(f"**Invoices with ≥ 4 failures:** {len(fail_counts[fail_counts['fail_count'] >= 4])}")
        st.markdown(f"**Total validation failures:** {int(fail_counts['fail_count'].sum())}")

        failures_by_analysis = (df[df['is_valid'] == False]
                                .groupby('analysis_name')
                                .size()
                                .reset_index(name='n_fallos')
                                .sort_values('n_fallos', ascending=False))
        failures_by_analysis['analysis_name'] = failures_by_analysis['analysis_name'].apply(translate)
        st.markdown("### Failure count by validation type")
        st.dataframe(failures_by_analysis, use_container_width=True)

        st.markdown("### Histogram: Validation Failures by Type")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(failures_by_analysis['analysis_name'], failures_by_analysis['n_fallos'], color='salmon')
        ax.set_title("Failed Validations by Type")
        ax.set_ylabel("Number of Failures")
        ax.set_xlabel("Validation Type")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

        st.markdown("### Histogram: Invoices by Number of Failures")
        failure_distribution = fail_counts['fail_count'].value_counts().sort_index()
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.bar(failure_distribution.index.astype(int), failure_distribution.values, color='skyblue')
        ax2.set_title("Failure Distribution per Invoice")
        ax2.set_xlabel("Number of Failures")
        ax2.set_ylabel("Number of Invoices")
        ax2.set_xticks(failure_distribution.index.astype(int))
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Critical error: {str(e)}")
        st.stop()
else:
    st.info("ℹ️ Upload a CSV file to start the analysis")
