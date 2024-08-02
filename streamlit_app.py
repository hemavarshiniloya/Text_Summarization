import streamlit as st
from txtai.pipeline import Summary
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, BartTokenizer, BartForConditionalGeneration
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

# List of languages with their ISO 639-1 codes
languages = {
    "English": "en", 
    "Afrikaans": "af",
    "Albanian": "sq",
    "Amharic": "am",
    "Arabic": "ar",
    "Armenian": "hy",
    "Azerbaijani": "az",
    "Basque": "eu",
    "Belarusian": "be",
    "Bengali": "bn",
    "Bosnian": "bs",
    "Bulgarian": "bg",
    "Catalan": "ca",
    "Chinese (Simplified)": "zh",
    "Chinese (Traditional)": "zh-TW",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "Esperanto": "eo",
    "Estonian": "et",
    "Finnish": "fi",
    "French": "fr",
    "Galician": "gl",
    "Georgian": "ka",
    "German": "de",
    "Greek": "el",
    "Gujarati": "gu",
    "Haitian Creole": "ht",
    "Hausa": "ha",
    "Hebrew": "he",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Icelandic": "is",
    "Igbo": "ig",
    "Indonesian": "id",
    "Irish": "ga",
    "Italian": "it",
    "Japanese": "ja",
    "Javanese": "jv",
    "Kannada": "kn",
    "Kazakh": "kk",
    "Khmer": "km",
    "Kinyarwanda": "rw",
    "Korean": "ko",
    "Kurdish": "ku",
    "Kyrgyz": "ky",
    "Lao": "lo",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Luxembourgish": "lb",
    "Macedonian": "mk",
    "Malagasy": "mg",
    "Malay": "ms",
    "Malayalam": "ml",
    "Maltese": "mt",
    "Maori": "mi",
    "Marathi": "mr",
    "Mongolian": "mn",
    "Nepali": "ne",
    "Norwegian": "no",
    "Pashto": "ps",
    "Persian": "fa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Punjabi": "pa",
    "Romanian": "ro",
    "Russian": "ru",
    "Samoan": "sm",
    "Scots Gaelic": "gd",
    "Serbian": "sr",
    "Sesotho": "st",
    "Shona": "sn",
    "Sindhi": "sd",
    "Sinhala": "si",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Somali": "so",
    "Spanish": "es",
    "Sundanese": "su",
    "Swahili": "sw",
    "Swedish": "sv",
    "Tagalog": "tl",
    "Tajik": "tg",
    "Tamil": "ta",
    "Tatar": "tt",
    "Telugu": "te",
    "Thai": "th",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Uzbek": "uz",
    "Vietnamese": "vi",
    "Welsh": "cy",
    "Xhosa": "xh",
    "Yoruba": "yo",
    "Zulu": "zu"
}

# Set page configuration
st.set_page_config(layout="wide")

# Initialize text summarizer
def text_summary(text, maxlength=None):
    summary = Summary()
    result = summary(text)
    return result

# Initialize tokenizer and model for sentiment analysis
def initialize_sentiment_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
        return tokenizer, model
    except Exception as e:
        st.error(f"An error occurred while loading the sentiment model: {str(e)}")
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
        st.error(f"An error occurred during sentiment analysis: {str(e)}")
        return {"label": "ERROR", "score": 0.0}

# Function to preprocess text
def preprocess_text(text):
    # Remove extra whitespace and special characters
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
        st.error(f"An error occurred: {str(e)}")
        return None

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Function to extract text from TXT
def extract_text_from_txt(file):
    return file.read().decode("utf-8")

# Function to extract text from HTML
def extract_text_from_html(file):
    soup = BeautifulSoup(file.read(), "html.parser")
    paragraphs = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
    text = "\n".join([p.get_text() for p in paragraphs])
    return text

# Function to extract text from CSV
def extract_text_from_csv(file):
    df = pd.read_csv(file)
    return df.to_string()

# Function to extract text from XML
def extract_text_from_xml(file):
    tree = ET.parse(file)
    root = tree.getroot()
    text = " ".join([elem.text for elem in root.iter() if elem.text])
    return text

# Function to extract text from Image
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
            # Fallback if UTF-8 decoding fails
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
    elif choice == "Generate Questions":
        st.session_state.text = ""

# Function to validate input
def validate_input(text):
    return bool(text and text.strip())

# Function to translate text using Google Translate API
def translate_text(text, target_language):
    translator = Translator()
    translated = translator.translate(text, dest=target_language)
    return translated.text

# Function to generate questions
def generate_questions(text):
    try:
        # Use a different model for question generation
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        
        inputs = tokenizer("generate questions: " + text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs['input_ids'], max_length=150, num_beams=5, early_stopping=True)
        questions = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return questions
    except Exception as e:
        st.error(f"An error occurred while generating questions: {str(e)}")
        return "Error generating questions."

# Function to download file
def download_file(content, filename):
    st.download_button(label="Download Summary", data=content, file_name=filename, mime="text/plain")

# Main function to run the Streamlit app
def main():
    st.title("Text Summarization and Sentiment Analysis App")

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

    # Initialize sentiment model
    tokenizer, model = initialize_sentiment_model()

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
                        elif uploaded_file.type == "text/csv":
                            text += extract_text_from_csv(uploaded_file)
                        elif uploaded_file.type == "application/xml":
                            text += extract_text_from_xml(uploaded_file)
                        elif uploaded_file.type.startswith("image/"):
                            text += extract_text_from_image(uploaded_file)
                    
                    summary = text_summary(text)
                    translated_summary = translate_text(summary, languages[selected_language])
                    
                    # Display summary
                    st.write("### Summary")
                    st.write(translated_summary)

                    save_summary(translated_summary)
                    download_file(translated_summary, "summary.txt")

    elif choice == "Summarize Text from Clipboard":
        st.session_state.clipboard_text = st.text_area("Paste Text from Clipboard", st.session_state.clipboard_text)

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
                with st.spinner("Generating questions..."):
                    text = preprocess_text(st.session_state.text)
                    questions = generate_questions(text)
                    
                    # Display questions
                    st.write("### Generated Questions")
                    st.write(questions)

                    save_summary(questions)
                    download_file(questions, "questions.txt")

    if st.sidebar.button("Clear Input"):
        clear_input(choice)

    if st.sidebar.button("Clear Summary History"):
        clear_summary_history()

    if st.sidebar.button("Load Summary History"):
        history = load_summary_history()
        if history:
            st.write("### Summary History")
            st.write(history)
        else:
            st.write("No history available.")

if __name__ == "__main__":
    main()
