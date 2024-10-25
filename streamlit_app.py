import streamlit as st
import nltk
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import xml.etree.ElementTree as ET
import re

# Ensure necessary NLTK corpora are downloaded
nltk.download("punkt")

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

def summarize_text(text, num_sentences=3):
    """Summarize the given text using TextBlob."""
    blob = TextBlob(text)
    sentences = blob.sentences
    # Extract sentences with the most noun phrases as a simple summarization
    sorted_sentences = sorted(sentences, key=lambda s: len(s.noun_phrases), reverse=True)
    summary = ' '.join(str(sentence) for sentence in sorted_sentences[:num_sentences])
    return summary

def main():
    st.title("Text Summarization App")

    # Handle choice selection
    choice = st.sidebar.radio("Choose an option", ["Summarize Text", "Summarize URL", "Summarize Document"])

    # Text Summarization Section
    if choice == "Summarize Text":
        input_text = st.text_area("Enter your text here:", height=300)
        num_sentences = st.number_input("Number of sentences for summary:", min_value=1, value=3, step=1)

        if st.button("Summarize"):
            if input_text:
                preprocessed_text = preprocess_text(input_text)
                summary = summarize_text(preprocessed_text, num_sentences)
                st.write("### Summary")
                st.write(summary)
            else:
                st.error("Please enter valid text.")

    # URL Summarization Section
    elif choice == "Summarize URL":
        url = st.text_input("Enter URL:")
        num_sentences = st.number_input("Number of sentences for summary:", min_value=1, value=3, step=1)

        if st.button("Summarize"):
            if url:
                text_from_url = extract_text_from_url(url)
                if text_from_url:
                    summary = summarize_text(text_from_url, num_sentences)
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
                file_type = uploaded_file.type
                if file_type == "application/pdf":
                    all_text = extract_text_from_pdf(uploaded_file)
                elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    all_text = extract_text_from_docx(uploaded_file)
                elif file_type == "text/plain":
                    all_text = extract_text_from_txt(uploaded_file)
                elif file_type == "text/html":
                    all_text = extract_text_from_html(uploaded_file)
                elif file_type == "text/csv":
                    all_text = extract_text_from_csv(uploaded_file)
                elif file_type == "application/xml":
                    all_text = extract_text_from_xml(uploaded_file)

                if all_text:
                    num_sentences = st.number_input("Number of sentences for summary:", min_value=1, value=3, step=1)
                    summary = summarize_text(all_text, num_sentences)
                    st.write("### Summary")
                    st.write(summary)
                else:
                    st.error("No text found in the uploaded document.")
            else:
                st.error("Please upload a document.")

if __name__ == "__main__":
    main()
