import streamlit as st
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import xml.etree.ElementTree as ET
import re
from googletrans import Translator

# Set up the summarization pipeline with an explicit model
model_name = "sshleifer/distilbart-cnn-12-6"
summarizer = pipeline("summarization", model=model_name)

# Set page configuration
st.set_page_config(layout="wide")

def preprocess_text(text):
    """Preprocess the input text."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()

def translate_text(text, target_lang='en'):
    """Translate the input text to the target language."""
    translator = Translator()
    try:
        translated = translator.translate(text, dest=target_lang)
        return translated.text
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text  # Return original text on failure

def download_summary(summary_text, filename='summary.txt'):
    """Download the summarized text."""
    st.download_button("Download Summary", data=summary_text, file_name=filename, mime='text/plain')

def main():
    st.title("Text Summarization App")

    # Language Selection for output summary
    language_options = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "zh-CN": "Chinese (Simplified)",
        "zh-TW": "Chinese (Traditional)",
        "ru": "Russian",
        "ja": "Japanese",
        "ko": "Korean",
        "ar": "Arabic",
        "hi": "Hindi",
        "tr": "Turkish",
        "nl": "Dutch",
        "sv": "Swedish",
        "fi": "Finnish",
        "da": "Danish",
        "no": "Norwegian",
        "th": "Thai",
        "vi": "Vietnamese",
        "tl": "Filipino",
        "ms": "Malay",
        "sw": "Swahili",
        "iw": "Hebrew",
        "bn": "Bengali",
        "pa": "Punjabi",
        "gu": "Gujarati",
        "ta": "Tamil",
        "te": "Telugu",
        "ml": "Malayalam",
        "mr": "Marathi",
        "kn": "Kannada",
        "si": "Sinhalese",
        "ur": "Urdu",
        "el": "Greek",
        "hu": "Hungarian",
        "cs": "Czech",
        "ro": "Romanian",
        "sl": "Slovenian",
        "hr": "Croatian",
        "sk": "Slovak",
        "bg": "Bulgarian",
        "sr": "Serbian",
        "et": "Estonian",
        "lv": "Latvian",
        "lt": "Lithuanian",
        "is": "Icelandic",
        "af": "Afrikaans",
        "sw": "Swahili"
    }

    # Display language selection
    target_lang = st.selectbox("Select target language for summary:", list(language_options.keys()), format_func=lambda x: language_options[x])

    # Handle choice selection
    choice = st.sidebar.radio("Choose an option", ["Summarize Text", "Summarize URL", "Summarize Document"])

    # Text Summarization Section
    if choice == "Summarize Text":
        input_text = st.text_area("Enter your text here:", height=300)
        max_length = st.number_input("Max length of summary (tokens):", min_value=1, value=150, step=1)
        min_length = st.number_input("Min length of summary (tokens):", min_value=1, value=30, step=1)

        if st.button("Summarize"):
            if input_text:
                preprocessed_text = preprocess_text(input_text)
                # Summarize the text first
                summary = summarizer(preprocessed_text, max_length=max_length, min_length=min_length, do_sample=False)
                summary_text = summary[0]['summary_text']
                
                # Translate the summary if the target language is not English
                if target_lang != 'en':
                    summary_text = translate_text(summary_text, target_lang)

                st.write("### Summary")
                st.write(summary_text)
                download_summary(summary_text)
            else:
                st.error("Please enter valid text.")

    # URL Summarization Section
    elif choice == "Summarize URL":
        url = st.text_input("Enter URL:")
        if st.button("Summarize"):
            if url:
                text_from_url = extract_text_from_url(url)
                if text_from_url:
                    summary = summarizer(text_from_url, max_length=150, min_length=30, do_sample=False)
                    summary_text = summary[0]['summary_text']
                    
                    # Translate the summary if the target language is not English
                    if target_lang != 'en':
                        summary_text = translate_text(summary_text, target_lang)

                    st.write("### Summary")
                    st.write(summary_text)
                    download_summary(summary_text)
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
                # Extract text based on file type
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
                    summary = summarizer(all_text, max_length=150, min_length=30, do_sample=False)
                    summary_text = summary[0]['summary_text']
                    
                    # Translate the summary if the target language is not English
                    if target_lang != 'en':
                        summary_text = translate_text(summary_text, target_lang)

                    st.write("### Summary")
                    st.write(summary_text)
                    download_summary(summary_text)
                else:
                    st.error("No text found in the uploaded document.")
            else:
                st.error("Please upload a document.")

if __name__ == "__main__":
    main()
