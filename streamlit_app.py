import streamlit as st
import PyPDF2
import docx
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from collections import Counter
from transformers import pipeline

class TextSummarizer:
    def __init__(self, language='english'):
        self.language = language
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn" if language == 'english' else "google/mt5-small")

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.]', '', text)
        return text
    
    def get_sentences(self, text):
        sentences = re.split(r'[.!?]+', text)
        return [sent.strip() for sent in sentences if sent.strip()]
    
    def summarize(self, text, num_sentences=3):
        try:
            cleaned_text = self.preprocess_text(text)
            summary = self.summarizer(cleaned_text, max_length=130, min_length=30, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            return f"An error occurred while summarizing: {str(e)}"

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def extract_text_from_docx(docx_file):
    try:
        doc = docx.Document(docx_file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {str(e)}")
        return None

def extract_text_from_txt(txt_file):
    try:
        text = txt_file.read().decode("utf-8")
        return text
    except Exception as e:
        st.error(f"Error extracting text from TXT: {str(e)}")
        return None

def extract_text_from_excel(excel_file):
    try:
        df = pd.read_excel(excel_file, sheet_name=None)  # Read all sheets
        text = ""
        for sheet_name, sheet_data in df.items():
            text += f"Sheet: {sheet_name}\n"
            text += sheet_data.to_string(index=False) + "\n\n"  # Convert DataFrame to string
        return text
    except Exception as e:
        st.error(f"Error extracting text from Excel: {str(e)}")
        return None

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = "\n".join(paragraph.get_text() for paragraph in paragraphs)
        return text
    except Exception as e:
        st.error(f"Error extracting text from URL: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Summarizer",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Summarizer")
    st.write("Upload multiple documents (PDF, DOCX, TXT, XLSX), input text directly, or enter a URL to generate summaries.")

    language = st.selectbox("Select Language for Summarization:", ["english", "spanish", "french", "german"])  # Add more languages as needed
    summarizer = TextSummarizer(language)

    uploaded_files = st.file_uploader("Upload documents (PDF, DOCX, TXT, XLSX)", type=['pdf', 'docx', 'txt', 'xlsx'], accept_multiple _files=True)
    input_text = st.text_area("Input text directly:")
    input_url = st.text_input("Enter a URL:")

    num_sentences = st.slider("Number of sentences in summary:", min_value=1, max_value=10, value=3)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.subheader(f"Processing {uploaded_file.name} ...")
            
            if uploaded_file.type == "application/pdf":
                text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = extract_text_from_docx(uploaded_file)
            elif uploaded_file.type == "text/plain":
                text = extract_text_from_txt(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                text = extract_text_from_excel(uploaded_file)
            else:
                st.error(f"Unsupported file type: {uploaded_file.type}")
                continue
            
            if text:
                summary = summarizer.summarize(text, num_sentences)
                st.write(f"**Summary of {uploaded_file.name}**: {summary}")
            else:
                st.write(f"**Error processing {uploaded_file.name}**")

    if input_text:
        st.subheader("Processing input text ...")
        summary = summarizer.summarize(input_text, num_sentences)
        st.write(f"**Summary of input text**: {summary}")

    if input_url:
        st.subheader("Processing URL ...")
        text = extract_text_from_url(input_url)
        if text:
            summary = summarizer.summarize(text, num_sentences)
            st.write(f"**Summary of {input_url}**: {summary}")
        else:
            st.write(f"**Error processing {input_url}**")

if __name__ == "__main__":
    main()
