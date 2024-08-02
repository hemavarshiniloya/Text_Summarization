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
    result = summary(text, max_length=maxlength)
    return result

# Initialize tokenizer and model for sentiment analysis
def initialize_sentiment_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
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

# Function to generate questions from text
def generate_questions(text):
    try:
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        
        inputs = tokenizer("generate questions: " + text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs['input_ids'], max_length=150, num_beams=5, early_stopping=True)
        questions = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return questions
    except Exception as e:
        st.error(f"An error occurred while generating questions: {str(e)}")
        return "Error generating questions."

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

# Clear input fields
def clear_input(choice):
    if choice == "Clear Input":
        st.session_state.text_input = ""
    elif choice == "Clear All":
        clear_summary_history()
        st.session_state.text_input = ""

# Define Streamlit UI
st.title("Text Analysis Tool")

st.sidebar.title("Options")
choice = st.sidebar.selectbox("Select an option", ["Summarize", "Sentiment Analysis", "Generate Questions", "Extract Text", "Language Translation", "Clear Input/All"])

if choice == "Summarize":
    st.header("Text Summarization")
    text_input = st.text_area("Enter text to summarize", height=200)
    maxlength = st.number_input("Maximum length for summary", min_value=50, max_value=2000, value=150)
    
    if st.button("Summarize"):
        summary = text_summary(text_input, maxlength)
        st.write("Summary:")
        st.write(summary)
        save_summary(summary)
        
elif choice == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    text_input = st.text_area("Enter text for sentiment analysis", height=200)
    
    tokenizer, model = initialize_sentiment_model()
    
    if tokenizer and model:
        if st.button("Analyze Sentiment"):
            result = sentiment_analysis(text_input, tokenizer, model)
            st.write("Sentiment:")
            st.write(result["label"])
            st.write("Score:")
            st.write(result["score"])

elif choice == "Generate Questions":
    st.header("Question Generation")
    text_input = st.text_area("Enter text to generate questions", height=200)
    
    if st.button("Generate Questions"):
        questions = generate_questions(text_input)
        st.write("Generated Questions:")
        st.write(questions)

elif choice == "Extract Text":
    st.header("Text Extraction")
    upload_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt", "html", "csv", "xml", "png", "jpg", "jpeg"])
    
    if upload_file is not None:
        file_type = upload_file.type.split('/')[1]
        if file_type in ["pdf", "docx", "txt"]:
            if file_type == "pdf":
                text = extract_text_from_pdf(upload_file)
            elif file_type == "docx":
                text = extract_text_from_docx(upload_file)
            else:
                text = extract_text_from_txt(upload_file)
        elif file_type == "html":
            text = extract_text_from_html(upload_file)
        elif file_type == "csv":
            text = extract_text_from_csv(upload_file)
        elif file_type == "xml":
            text = extract_text_from_xml(upload_file)
        elif file_type in ["png", "jpg", "jpeg"]:
            text = extract_text_from_image(upload_file)
        else:
            st.error("Unsupported file type")
            text = ""
        
        st.write("Extracted Text:")
        st.write(text)
        
elif choice == "Language Translation":
    st.header("Language Translation")
    text_input = st.text_area("Enter text to translate", height=200)
    target_language = st.selectbox("Select target language", list(languages.keys()))
    
    if st.button("Translate"):
        translator = Translator()
        target_lang_code = languages[target_language]
        translated = translator.translate(text_input, dest=target_lang_code).text
        st.write("Translated Text:")
        st.write(translated)

elif choice == "Clear Input/All":
    st.header("Clear Input/All")
    clear_choice = st.selectbox("Choose what to clear", ["Clear Input", "Clear All"])
    
    if st.button("Clear"):
        clear_input(clear_choice)
        st.success(f"{clear_choice} cleared!")

# Display summary history
st.sidebar.title("Summary History")
history = load_summary_history()
st.sidebar.text_area("History", history, height=300)
