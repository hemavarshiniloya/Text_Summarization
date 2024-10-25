import streamlit as st
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import os
import re

# Set page configuration
st.set_page_config(layout="wide")

def text_summary(text, ratio):
    """Summarize the given text using Gensim's summarize."""
    if not text:
        raise ValueError("Input text cannot be empty.")
    return summarize(text, ratio=ratio)

def preprocess_text(text):
    """Preprocess the input text."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text.strip()

def extract_text_from_url(url):
    """Extract text from a URL."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
        text = "\n".join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        st.error(f"An error occurred while extracting text from URL: {str(e)}")
        return ""

def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file):
    """Extract text from a DOCX file."""
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_txt(file):
    """Extract text from a TXT file."""
    return file.read().decode("utf-8")

def extract_text_from_html(file):
    """Extract text from an HTML file."""
    soup = BeautifulSoup(file.read(), "html.parser")
    paragraphs = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
    text = "\n".join([p.get_text() for p in paragraphs])
    return text

def extract_text_from_csv(file):
    """Extract text from a CSV file."""
    df = pd.read_csv(file)
    return df.to_string()

def extract_text_from_xml(file):
    """Extract text from an XML file."""
    tree = ET.parse(file)
    root = tree.getroot()
    text = " ".join([elem.text for elem in root.iter() if elem.text])
    return text

def download_file(content, filename):
    """Download file functionality."""
    st.download_button(label="Download Summary", data=content, file_name=filename, mime="text/plain")

def main():
    st.title("Text Summarization App")

    # Initialize session state attributes if they don't exist
    if 'text' not in st.session_state:
        st.session_state.text = ""
    if 'url' not in st.session_state:
        st.session_state.url = ""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

    # Handle choice selection
    choice = st.sidebar.radio("Choose an option", ["Summarize Text", "Summarize URL", "Summarize Document"])

    # Text Summarization Section
    if choice == "Summarize Text":
        st.session_state.text = st.text_area("Enter your text here:", value=st.session_state.text, height=300)
        ratio = st.number_input("Summary ratio (0.0 to 1.0)", min_value=0.0, max_value=1.0, value=0.2, step=0.01)

        if st.button("Summarize"):
            if st.session_state.text:
                preprocessed_text = preprocess_text(st.session_state.text)
                try:
                    summary = text_summary(preprocessed_text, ratio)
                    st.write("### Summary")
                    st.write(summary)

                    # Download summary
                    download_file(summary, "summary.txt")
                except Exception as e:
                    st.error(f"An error occurred during summarization: {str(e)}")
            else:
                st.error("Please enter valid text.")

    # URL Summarization Section
    elif choice == "Summarize URL":
        st.session_state.url = st.text_input("Enter URL:", value=st.session_state.url)

        if st.button("Summarize"):
            if st.session_state.url:
                text_from_url = extract_text_from_url(st.session_state.url)
                if text_from_url:
                    try:
                        summary = text_summary(text_from_url, 0.2)
                        st.write("### Summary")
                        st.write(summary)

                        # Download summary
                        download_file(summary, "summary.txt")
                    except Exception as e:
                        st.error(f"An error occurred during summarization: {str(e)}")
                else:
                    st.error("No text found at the provided URL.")
            else:
                st.error("Please enter a valid URL.")

    # Document Summarization Section
    elif choice == "Summarize Document":
        uploaded_files = st.file_uploader("Choose a file (PDF, DOCX, TXT, HTML, CSV, XML)", accept_multiple_files=True)

        if st.button("Summarize"):
            if uploaded_files:
                all_text = ""
                for uploaded_file in uploaded_files:
                    if uploaded_file.type == "application/pdf":
                        all_text += extract_text_from_pdf(uploaded_file)
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        all_text += extract_text_from_docx(uploaded_file)
                    elif uploaded_file.type == "text/plain":
                        all_text += extract_text_from_txt(uploaded_file)
                    elif uploaded_file.type == "text/html":
                        all_text += extract_text_from_html(uploaded_file)
                    elif uploaded_file.type == "text/csv":
                        all_text += extract_text_from_csv(uploaded_file)
                    elif uploaded_file.type == "application/xml":
                        all_text += extract_text_from_xml(uploaded_file)

                if all_text:
                    try:
                        summary = text_summary(all_text, 0.2)
                        st.write("### Summary")
                        st.write(summary)

                        # Download summary
                        download_file(summary, "summary.txt")
                    except Exception as e:
                        st.error(f"An error occurred during summarization: {str(e)}")
                else:
                    st.error("No text found in the uploaded documents.")
            else:
                st.error("Please upload at least one document.")

if __name__ == "__main__":
    main()
