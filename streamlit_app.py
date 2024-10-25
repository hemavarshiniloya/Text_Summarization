import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import xml.etree.ElementTree as ET
import os
import re

# Ensure NLTK data is downloaded
def ensure_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)  # Download 'punkt' quietly

# Set page configuration
st.set_page_config(layout="wide")

def text_summary(text, max_sentences):
    """Summarize the given text using LSA summarizer."""
    if not text:
        raise ValueError("Input text cannot be empty.")

    # Ensure NLTK data is available
    ensure_nltk_data()

    # Use the Tokenizer after ensuring NLTK data is present
    tokenizer = Tokenizer("english")
    parser = PlaintextParser.from_string(text, tokenizer)
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, max_sentences)
    return " ".join(str(sentence) for sentence in summary)

   # Ensure NLTK data is downloaded
   def ensure_nltk_data():
       """Ensure NLTK tokenizer resources are downloaded."""
       try:
           nltk.data.find("tokenizers/punkt")
       except LookupError:
           nltk.download("punkt", quiet=True)  # Download 'punkt' quietly

def preprocess_text(text):
    """Preprocess the input text."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
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

def save_summary(summary):
    """Save summary to history."""
    filename = "summary_history.txt"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(summary + "\n\n")

def load_summary_history():
    """Load summary history."""
    filename = "summary_history.txt"
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def clear_summary_history():
    """Clear summary history."""
    filename = "summary_history.txt"
    if os.path.exists(filename):
        os.remove(filename)

def clear_input(choice):
    """Clear input fields based on choice."""
    if choice == "Summarize Text":
        st.session_state.text = ""
    elif choice == "Summarize URL":
        st.session_state.url = ""
    elif choice == "Summarize Document":
        st.session_state.uploaded_files = []
    elif choice == "Summarize Text from Clipboard":
        st.session_state.clipboard_text = ""

def validate_input(text):
    """Validate input text."""
    return bool(text and text.strip())

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
    if 'clipboard_text' not in st.session_state:
        st.session_state.clipboard_text = ""

    # Ensure NLTK data is available
    ensure_nltk_data()

    # Handle choice selection
    choice = st.sidebar.radio("Choose an option", ["Summarize Text", "Summarize URL", "Summarize Document", "Summarize Text from Clipboard"])

    # Text Summarization Section
    if choice == "Summarize Text":
        st.session_state.text = st.text_area("Enter your text here:", value=st.session_state.text, height=300)
        max_sentences = st.number_input("Maximum number of sentences for summary", min_value=1, value=2, step=1)

        if st.button("Summarize"):
            if validate_input(st.session_state.text):
                preprocessed_text = preprocess_text(st.session_state.text)
                try:
                    summary = text_summary(preprocessed_text, max_sentences)
                    st.write("### Summary")
                    st.write(summary)

                    # Save summary to history
                    save_summary(summary)

                    # Download summary
                    download_file(summary, "summary.txt")
                except Exception as e:
                    st.error(f"An error occurred during summarization: {str(e)}")
            else:
                st.error("Please enter valid text.")

        if st.button("Clear Input"):
            clear_input("Summarize Text")

    # URL Summarization Section
    elif choice == "Summarize URL":
        st.session_state.url = st.text_input("Enter URL:", value=st.session_state.url)

        if st.button("Summarize"):
            if st.session_state.url:
                text_from_url = extract_text_from_url(st.session_state.url)
                if validate_input(text_from_url):
                    try:
                        summary = text_summary(text_from_url, 2)
                        st.write("### Summary")
                        st.write(summary)

                        # Save summary to history
                        save_summary(summary)

                        # Download summary
                        download_file(summary, "summary.txt")
                    except Exception as e:
                        st.error(f"An error occurred during summarization: {str(e)}")
                else:
                    st.error("No text found at the provided URL.")
            else:
                st.error("Please enter a valid URL.")

        if st.button("Clear Input"):
            clear_input("Summarize URL")

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

                if validate_input(all_text):
                    try:
                        summary = text_summary(all_text, 2)
                        st.write("### Summary")
                        st.write(summary)

                        # Save summary to history
                        save_summary(summary)

                        # Download summary
                        download_file(summary, "summary.txt")
                    except Exception as e:
                        st.error(f"An error occurred during summarization: {str(e)}")
                else:
                    st.error("No text found in the uploaded documents.")
            else:
                st.error("Please upload at least one document.")

        if st.button("Clear Input"):
            clear_input("Summarize Document")

    # Clipboard Summarization Section
    elif choice == "Summarize Text from Clipboard":
        st.session_state.clipboard_text = st.text_area("Paste text from clipboard here:", value=st.session_state.clipboard_text, height=300)

        if st.button("Summarize"):
            if validate_input(st.session_state.clipboard_text):
                preprocessed_text = preprocess_text(st.session_state.clipboard_text)
                try:
                    summary = text_summary(preprocessed_text, 2)
                    st.write("### Summary")
                    st.write(summary)

                    # Save summary to history
                    save_summary(summary)

                    # Download summary
                    download_file(summary, "summary.txt")
                except Exception as e:
                    st.error(f"An error occurred during summarization: {str(e)}")
            else:
                st.error("Please paste valid text.")

        if st.button("Clear Input"):
            clear_input("Summarize Text from Clipboard")

    # Summary History Section
    if st.sidebar.button("Show Summary History"):
        history = load_summary_history()
        if history:
            st.sidebar.text_area("Summary History", history, height=300)
        else:
            st.sidebar.write("No summary history available.")

    if st.sidebar.button("Clear Summary History"):
        clear_summary_history()
        st.sidebar.success("Summary history cleared.")

if __name__ == "__main__":
    main()
