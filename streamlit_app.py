import streamlit as st
from txtai.pipeline import Summary
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BartTokenizer, BartForConditionalGeneration
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import xml.etree.ElementTree as ET
import os
import pytesseract
from googletrans import Translator
import re
import torch
from PIL import Image
import logging

# Set up logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Print versions for debugging
st.sidebar.write(f"transformers version: {transformers.__version__}")
st.sidebar.write(f"torch version: {torch.__version__}")
st.sidebar.write(f"streamlit version: {st.__version__}")

# List of languages
languages = {
    # Add your languages here
}

# Set page configuration
st.set_page_config(layout="wide")

# Initialize text summarizer
def text_summary(text, maxlength=None):
    try:
        summary = Summary()
        result = summary(text)
        return result
    except Exception as e:
        logging.error(f"Error in text summarization: {str(e)}")
        st.error("An error occurred during text summarization.")
        return ""

# Initialize tokenizer and model for sentiment analysis
def initialize_sentiment_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
        return tokenizer, model
    except Exception as e:
        logging.error(f"Error loading sentiment model: {str(e)}")
        st.error("An error occurred while loading the sentiment model.")
        return None, None

# Perform sentiment analysis
def sentiment_analysis(text, tokenizer, model):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        return {"label": "POSITIVE" if predictions.item() == 1 else "NEGATIVE", "score": torch.softmax(logits, dim=1)[0, predictions.item()].item()}
    except Exception as e:
        logging.error(f"Error during sentiment analysis: {str(e)}")
        st.error("An error occurred during sentiment analysis.")
        return {"label": "ERROR", "score": 0.0}

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# Function to extract text from URL
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
        text = "\n".join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        logging.error(f"Error extracting text from URL: {str(e)}")
        st.error("An error occurred while extracting text from the URL.")
        return None

# Function to extract text from files
def extract_text_from_file(file, file_type):
    try:
        if file_type == "application/pdf":
            return extract_text_from_pdf(file)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return extract_text_from_docx(file)
        elif file_type == "text/plain":
            return extract_text_from_txt(file)
        elif file_type == "text/html":
            return extract_text_from_html(file)
        elif file_type == "text/csv":
            return extract_text_from_csv(file)
        elif file_type == "application/xml":
            return extract_text_from_xml(file)
        elif file_type.startswith("image/"):
            return extract_text_from_image(file)
        else:
            st.warning(f"Unsupported file type: {file_type}")
            return ""
    except Exception as e:
        logging.error(f"Error extracting text from file: {str(e)}")
        st.error("An error occurred while extracting text from the file.")
        return ""

# Functions to extract text from specific file formats
def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {str(e)}")
        st.error("An error occurred while extracting text from the PDF.")
        return ""

def extract_text_from_docx(file):
    try:
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        logging.error(f"Error extracting text from DOCX: {str(e)}")
        st.error("An error occurred while extracting text from the DOCX file.")
        return ""

def extract_text_from_txt(file):
    try:
        return file.read().decode("utf-8")
    except Exception as e:
        logging.error(f"Error extracting text from TXT: {str(e)}")
        st.error("An error occurred while extracting text from the TXT file.")
        return ""

def extract_text_from_html(file):
    try:
        soup = BeautifulSoup(file.read(), "html.parser")
        paragraphs = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
        text = "\n".join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        logging.error(f"Error extracting text from HTML: {str(e)}")
        st.error("An error occurred while extracting text from the HTML file.")
        return ""

def extract_text_from_csv(file):
    try:
        df = pd.read_csv(file)
        return df.to_string()
    except Exception as e:
        logging.error(f"Error extracting text from CSV: {str(e)}")
        st.error("An error occurred while extracting text from the CSV file.")
        return ""

def extract_text_from_xml(file):
    try:
        tree = ET.parse(file)
        root = tree.getroot()
        text = " ".join([elem.text for elem in root.iter() if elem.text])
        return text
    except Exception as e:
        logging.error(f"Error extracting text from XML: {str(e)}")
        st.error("An error occurred while extracting text from the XML file.")
        return ""

def extract_text_from_image(file):
    try:
        image = Image.open(file)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        logging.error(f"Error extracting text from image: {str(e)}")
        st.error("An error occurred while extracting text from the image.")
        return ""

# Function to save summary to history
def save_summary(summary):
    try:
        filename = "summary_history.txt"
        with open(filename, "a", encoding="utf-8") as f:
            f.write(summary + "\n\n")
    except Exception as e:
        logging.error(f"Error saving summary to history: {str(e)}")
        st.error("An error occurred while saving the summary.")

# Function to load summary history
def load_summary_history():
    try:
        filename = "summary_history.txt"
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                return f.read()
        return "No summary history available."
    except Exception as e:
        logging.error(f"Error loading summary history: {str(e)}")
        st.error("An error occurred while loading the summary history.")
        return "No summary history available."

# Function to clear summary history
def clear_summary_history():
    try:
        filename = "summary_history.txt"
        if os.path.exists(filename):
            os.remove(filename)
    except Exception as e:
        logging.error(f"Error clearing summary history: {str(e)}")
        st.error("An error occurred while clearing the summary history.")

# Streamlit UI
st.title("Text Summarization and Analysis App")

# Define the options
options = [
    "Summarize Text",
    "Summarize URL",
    "Summarize Document",
    "Summarize Text from Clipboard",
    "Generate Questions",
    "Clear Input",
    "View Summary History",
    "Clear Summary History"
]

choice = st.selectbox("Select an option", options)

if choice == "Summarize Text":
    text = st.text_area("Enter text for summarization")
    if st.button("Summarize"):
        if validate_input(text):
            summary = text_summary(text)
            st.write("Summary:", summary)
            save_summary(summary)
        else:
            st.warning("Please enter text to summarize.")

elif choice == "Summarize URL":
    url = st.text_input("Enter URL")
    if st.button("Summarize"):
        if validate_input(url):
            text = extract_text_from_url(url)
            if text:
                summary = text_summary(text)
                st.write("Summary:", summary)
                save_summary(summary)
        else:
            st.warning("Please enter a URL to summarize.")

elif choice == "Summarize Document":
    uploaded_files = st.file_uploader("Upload document", type=["pdf", "docx", "txt", "html", "csv", "xml", "jpg", "jpeg", "png"], accept_multiple_files=True)
    if st.button("Summarize"):
        if uploaded_files:
            text = ""
            for uploaded_file in uploaded_files:
                file_type = uploaded_file.type
                text += extract_text_from_file(uploaded_file, file_type)
            if text:
                summary = text_summary(text)
                st.write("Summary:", summary)
                save_summary(summary)
        else:
            st.warning("Please upload a document to summarize.")

elif choice == "Summarize Text from Clipboard":
    clipboard_text = st.text_area("Paste text from clipboard")
    if st.button("Summarize"):
        if validate_input(clipboard_text):
            summary = text_summary(clipboard_text)
            st.write("Summary:", summary)
            save_summary(summary)
        else:
            st.warning("Please paste text to summarize.")

elif choice == "Generate Questions":
    text = st.text_area("Enter text for question generation")
    if st.button("Generate Questions"):
        if validate_input(text):
            questions = generate_questions(text)
            st.write("Generated Questions:", questions)
        else:
            st.warning("Please enter text to generate questions.")

elif choice == "Clear Input":
    st.session_state.clear()
    st.success("Input fields cleared.")

elif choice == "View Summary History":
    history = load_summary_history()
    if history:
        st.write("Summary History:", history)
    else:
        st.write("No summary history available.")

elif choice == "Clear Summary History":
    clear_summary_history()
    st.success("Summary history cleared.")

# Display resource usage information
st.sidebar.header("Resource Usage")
st.sidebar.text("Monitor your system's memory and CPU usage to ensure that there are sufficient resources available. You can use tools like top, htop, or system monitoring utilities provided by your hosting environment.")

