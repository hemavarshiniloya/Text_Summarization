import streamlit as st
from gensim.summarization import summarize as gensim_summarize
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import xml.etree.ElementTree as ET
import re

# Set page configuration
st.set_page_config(layout="wide")

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

def extract_text_from_file(file):
    """Extract text from uploaded file based on its type."""
    if file.type == "application/pdf":
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text

    elif file.type == "text/plain":
        return file.read().decode("utf-8")

    elif file.type == "text/html":
        soup = BeautifulSoup(file.read(), "html.parser")
        paragraphs = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
        text = "\n".join([p.get_text() for p in paragraphs])
        return text

    elif file.type == "text/csv":
        df = pd.read_csv(file)
        return df.to_string()

    elif file.type == "application/xml":
        tree = ET.parse(file)
        root = tree.getroot()
        text = " ".join([elem.text for elem in root.iter() if elem.text])
        return text
    else:
        st.error("Unsupported file type.")
        return ""

def summarize_text(text, ratio=0.3):
    """Summarize the given text using Gensim's summarizer."""
    try:
        summary = gensim_summarize(text, ratio=ratio)
        return summary
    except ValueError:
        return "Text is too short for summarization."

def main():
    st.title("Enhanced Text Summarization App")

    # Handle choice selection
    choice = st.sidebar.radio("Choose an option", ["Summarize Text", "Summarize URL", "Summarize Document"])

    # Text Summarization Section
    if choice == "Summarize Text":
        input_text = st.text_area("Enter your text here:", height=300)
        ratio = st.slider("Summary Ratio (e.g., 0.1 for 10% of text):", min_value=0.1, max_value=1.0, value=0.3, step=0.1)

        if st.button("Summarize"):
            if input_text:
                preprocessed_text = preprocess_text(input_text)
                summary = summarize_text(preprocessed_text, ratio)
                st.write("### Summary")
                st.write(summary)
            else:
                st.error("Please enter valid text.")

    # URL Summarization Section
    elif choice == "Summarize URL":
        url = st.text_input("Enter URL:")
        ratio = st.slider("Summary Ratio (e.g., 0.1 for 10% of text):", min_value=0.1, max_value=1.0, value=0.3, step=0.1)

        if st.button("Summarize"):
            if url:
                text_from_url = extract_text_from_url(url)
                if text_from_url:
                    summary = summarize_text(text_from_url, ratio)
                    st.write("### Summary")
                    st.write(summary)
                else:
                    st.error("No text found at the provided URL.")
            else:
                st.error("Please enter a valid URL.")

    # Document Summarization Section
    elif choice == "Summarize Document":
        uploaded_file = st.file_uploader("Choose a file (PDF, DOCX, TXT, HTML, CSV, XML)", type=['pdf', 'docx', 'txt', 'html', 'csv', 'xml'])

        if st.button("Summarize"):
            if uploaded_file:
                all_text = extract_text_from_file(uploaded_file)
                if all_text:
                    ratio = st.slider("Summary Ratio (e.g., 0.1 for 10% of text):", min_value=0.1, max_value=1.0, value=0.3, step=0.1)
                    summary = summarize_text(all_text, ratio)
                    st.write("### Summary")
                    st.write(summary)
                else:
                    st.error("No text found in the uploaded document.")
            else:
                st.error("Please upload a document.")

if __name__ == "__main__":
    main()
