import streamlit as st
from txtai.pipeline import Summary
from transformers import pipeline
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
from PIL import Image

# Initialize question generator
def initialize_question_generator():
    return pipeline("text2text-generation", model="valhalla/t5-small-qa-qg-hl")

# List of languages with their ISO 639-1 codes
languages = {
    "English": "en", 
    # Add other languages as needed
}

# Set page configuration
st.set_page_config(layout="wide")

# Initialize text summarizer and question generator
def text_summary(text, maxlength=None):
    summary = Summary()
    result = summary(text)
    return result

def generate_questions(text):
    question_generator = initialize_question_generator()
    questions = question_generator(text, max_length=512, num_beams=4, early_stopping=True)
    return questions[0]['generated_text']

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# Function to extract text from various sources (URL, PDF, DOCX, etc.)
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
        text = "\n".join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_txt(file):
    return file.read().decode("utf-8")

def extract_text_from_html(file):
    soup = BeautifulSoup(file.read(), "html.parser")
    paragraphs = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
    text = "\n".join([p.get_text() for p in paragraphs])
    return text

def extract_text_from_csv(file):
    df = pd.read_csv(file)
    return df.to_string()

def extract_text_from_xml(file):
    tree = ET.parse(file)
    root = tree.getroot()
    text = " ".join([elem.text for elem in root.iter() if elem.text])
    return text

def extract_text_from_image(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return text

# Function to save summary to history
def save_summary(summary):
    filename = "summary_history.txt"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(summary + "\n\n")

# Function to load summary history
def load_summary_history():
    filename = "summary_history.txt"
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(filename, "r", encoding="latin1") as f:
                return f.read()
    return ""

# Function to clear summary history
def clear_summary_history():
    filename = "summary_history.txt"
    if os.path.exists(filename):
        os.remove(filename)

# Function to clear input fields based on choice
def clear_input(choice):
    if choice == "Summarize Text":
        st.session_state.text = ""
    elif choice == "Summarize URL":
        st.session_state.url = ""
    elif choice == "Summarize Document":
        st.session_state.uploaded_files = []
    elif choice == "Summarize Text from Clipboard":
        st.session_state.clipboard_text = ""

# Function to validate input
def validate_input(text):
    return bool(text and text.strip())

# Function to translate text using Google Translate API
def translate_text(text, target_language):
    translator = Translator()
    translated = translator.translate(text, dest=target_language)
    return translated.text

# Function to download file
def download_file(content, filename):
    st.download_button(label="Download Summary", data=content, file_name=filename, mime="text/plain")

# Main function to run the Streamlit app
def main():
    st.title("Text Summarization and Question Generation App")

    # Language selection
    selected_language = st.sidebar.selectbox("Select Language", options=list(languages.keys()), index=0)

    # Handle choice selection
    choice = st.sidebar.radio("Choose an option", ["Summarize Text", "Summarize URL", "Summarize Document", "Summarize Text from Clipboard", "Generate Questions"])

    # Initialize session state attributes if they don't exist
    if 'text' not in st.session_state:
        st.session_state.text = ""
    if 'url' not in st.session_state:
        st.session_state.url = ""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'clipboard_text' not in st.session_state:
        st.session_state.clipboard_text = ""

    # Handle each choice
    if choice == "Summarize Text":
        st.session_state.text = st.text_area("Enter Text", st.session_state.text)
        maxlength = st.slider("Maximum Summary Length", min_value=50, max_value=1000, value=200)

        if st.button("Summarize"):
            if validate_input(st.session_state.text):
                with st.spinner("Processing..."):
                    text = preprocess_text(st.session_state.text)
                    summary = text_summary(text, maxlength)
                    translated_summary = translate_text(summary, languages[selected_language])
                    
                    # Display summary
                    st.write("### Summary")
                    st.write(translated_summary)

                    save_summary(translated_summary)
                    download_file(translated_summary, "summary.txt")

    elif choice == "Summarize URL":
        st.session_state.url = st.text_input("Enter URL", st.session_state.url)

        if st.button("Summarize URL"):
            if validate_input(st.session_state.url):
                with st.spinner("Processing..."):
                    text = extract_text_from_url(st.session_state.url)
                    summary = text_summary(text)
                    translated_summary = translate_text(summary, languages[selected_language])
                    
                    # Display summary
                    st.write("### Summary")
                    st.write(translated_summary)

                    save_summary(translated_summary)
                    download_file(translated_summary, "summary.txt")

    elif choice == "Summarize Document":
        uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True)
        st.session_state.uploaded_files = uploaded_files

        if st.button("Summarize Document"):
            if uploaded_files:
                with st.spinner("Processing..."):
                    text = ""
                    for uploaded_file in uploaded_files:
                        if uploaded_file.type == "application/pdf":
                            text += extract_text_from_pdf(uploaded_file)
                        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                            text += extract_text_from_docx(uploaded_file)
                        elif uploaded_file.type == "text/plain":
                            text += extract_text_from_txt(uploaded_file)
                        elif uploaded_file.type == "text/html":
                            text += extract_text_from_html(uploaded_file)
                        elif uploaded_file.type == "application/xml":
                            text += extract_text_from_xml(uploaded_file)
                        elif uploaded_file.type == "image/jpeg" or uploaded_file.type == "image/png":
                            text += extract_text_from_image(uploaded_file)

                    summary = text_summary(text)
                    translated_summary = translate_text(summary, languages[selected_language])
                    
                    # Display summary
                    st.write("### Summary")
                    st.write(translated_summary)

                    save_summary(translated_summary)
                    download_file(translated_summary, "summary.txt")

    elif choice == "Summarize Text from Clipboard":
        st.session_state.clipboard_text = st.text_area("Paste Clipboard Text Here", st.session_state.clipboard_text)

        if st.button("Summarize Clipboard Text"):
            if validate_input(st.session_state.clipboard_text):
                with st.spinner("Processing..."):
                    text = preprocess_text(st.session_state.clipboard_text)
                    summary = text_summary(text)
                    translated_summary = translate_text(summary, languages[selected_language])
                    
                    # Display summary
                    st.write("### Summary")
                    st.write(translated_summary)

                    save_summary(translated_summary)
                    download_file(translated_summary, "summary.txt")

    elif choice == "Generate Questions":
        st.session_state.text = st.text_area("Enter Text for Question Generation", st.session_state.text)

        if st.button("Generate Questions"):
            if validate_input(st.session_state.text):
                with st.spinner("Processing..."):
                    text = preprocess_text(st.session_state.text)
                    questions = generate_questions(text)
                    
                    # Display questions
                    st.write("### Generated Questions")
                    st.write(questions)

                    save_summary(questions)
                    download_file(questions, "questions.txt")

    if st.sidebar.button("Clear Input"):
        clear_input(choice)

    if st.sidebar.button("Clear History"):
        clear_summary_history()
        st.sidebar.write("Summary history cleared.")

    # Display summary history
    st.sidebar.write("### Summary History")
    history = load_summary_history()
    st.sidebar.text_area("History", history, height=300)

if __name__ == "__main__":
    main()
