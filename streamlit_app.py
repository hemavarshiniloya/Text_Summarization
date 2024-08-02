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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import language_tool_python
import spacy

# Load Spacy model for keyword extraction
nlp = spacy.load("en_core_web_sm")

# Initialize text summarizer
def text_summary(text, maxlength=None):
    summary = Summary()
    result = summary(text)
    return result

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

# Function to extract text from Screenshot (image)
def extract_text_from_screenshot(file):
    return extract_text_from_image(file)

# Function to calculate text similarity
def calculate_text_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    similarity = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    return similarity

# Function to classify text
def classify_text(text, model_name='bert-base-uncased', labels=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_class = torch.argmax(probs, dim=-1).item()
    if labels:
        return labels[pred_class]
    return pred_class

# Function to extract keywords
def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.text for token in doc if token.is_stop == False and token.is_punct == False]
    return keywords

# Function to perform grammar and spell check
def grammar_and_spell_check(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    return matches

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
    st.title("Text Summarization and Feature Extraction App")

    # Language selection
    selected_language = st.sidebar.selectbox("Select Language", options=list(languages.keys()), index=0)

    # Handle choice selection
    choice = st.sidebar.radio("Choose an option", ["Summarize Text", "Summarize URL", "Summarize Document", "Summarize Text from Clipboard", "Generate Questions", "Text Similarity", "Classify Text", "Extract Keywords", "Grammar and Spell Check"])

    if "text" not in st.session_state:
        st.session_state.text = ""
    if "url" not in st.session_state:
        st.session_state.url = ""
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "clipboard_text" not in st.session_state:
        st.session_state.clipboard_text = ""
    if "text2" not in st.session_state:
        st.session_state.text2 = ""

    if choice == "Summarize Text":
        st.session_state.text = st.text_area("Enter Text", st.session_state.text)
        if st.button("Summarize"):
            if validate_input(st.session_state.text):
                text = preprocess_text(st.session_state.text)
                summary = text_summary(text)
                st.write("Summary:", summary)
                save_summary(summary)
                download_file(summary, "summary.txt")
            else:
                st.error("Please enter valid text.")

    elif choice == "Summarize URL":
        st.session_state.url = st.text_input("Enter URL", st.session_state.url)
        if st.button("Summarize"):
            if validate_input(st.session_state.url):
                text = extract_text_from_url(st.session_state.url)
                if text:
                    summary = text_summary(preprocess_text(text))
                    st.write("Summary:", summary)
                    save_summary(summary)
                    download_file(summary, "summary.txt")
                else:
                    st.error("Unable to extract text from the URL.")
            else:
                st.error("Please enter a valid URL.")

    elif choice == "Summarize Document":
        uploaded_files = st.file_uploader("Upload Document(s)", type=["pdf", "docx", "txt", "html", "csv", "xml", "jpg", "jpeg", "png"], accept_multiple_files=True)
        st.session_state.uploaded_files = uploaded_files
        if st.button("Summarize"):
            if st.session_state.uploaded_files:
                all_text = ""
                for uploaded_file in st.session_state.uploaded_files:
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
                    elif uploaded_file.type in ["image/jpeg", "image/png"]:
                        all_text += extract_text_from_image(uploaded_file)
                if all_text:
                    summary = text_summary(preprocess_text(all_text))
                    st.write("Summary:", summary)
                    save_summary(summary)
                    download_file(summary, "summary.txt")
                else:
                    st.error("Unable to extract text from the documents.")
            else:
                st.error("Please upload at least one document.")

    elif choice == "Summarize Text from Clipboard":
        st.session_state.clipboard_text = st.text_area("Enter Text from Clipboard", st.session_state.clipboard_text)
        if st.button("Summarize"):
            if validate_input(st.session_state.clipboard_text):
                text = preprocess_text(st.session_state.clipboard_text)
                summary = text_summary(text)
                st.write("Summary:", summary)
                save_summary(summary)
                download_file(summary, "summary.txt")
            else:
                st.error("Please enter valid text from clipboard.")


    elif choice == "Text Similarity":
        st.session_state.text = st.text_area("Enter First Text", st.session_state.text)
        st.session_state.text2 = st.text_area("Enter Second Text", st.session_state.text2)
        if st.button("Compare"):
            if validate_input(st.session_state.text) and validate_input(st.session_state.text2):
                similarity = calculate_text_similarity(st.session_state.text, st.session_state.text2)
                st.write(f"Similarity Score: {similarity:.2f}")
            else:
                st.error("Please enter valid texts.")

    elif choice == "Classify Text":
        st.session_state.text = st.text_area("Enter Text", st.session_state.text)
        if st.button("Classify"):
            if validate_input(st.session_state.text):
                labels = ["Category 1", "Category 2", "Category 3"]  # Define your categories here
                classification = classify_text(st.session_state.text, labels=labels)
                st.write("Classified as:", classification)
            else:
                st.error("Please enter valid text.")

    elif choice == "Extract Keywords":
        st.session_state.text = st.text_area("Enter Text", st.session_state.text)
        if st.button("Extract Keywords"):
            if validate_input(st.session_state.text):
                keywords = extract_keywords(st.session_state.text)
                st.write("Keywords:", keywords)
            else:
                st.error("Please enter valid text.")

    elif choice == "Grammar and Spell Check":
        st.session_state.text = st.text_area("Enter Text", st.session_state.text)
        if st.button("Check"):
            if validate_input(st.session_state.text):
                matches = grammar_and_spell_check(st.session_state.text)
                if matches:
                    st.write("Grammar and Spell Check Results:")
                    for match in matches:
                        st.write(f"{match.message} (at position {match.offset})")
                else:
                    st.write("No issues found.")
            else:
                st.error("Please enter valid text.")

    # Clear history
    if st.sidebar.button("Clear Summary History"):
        clear_summary_history()
        st.success("Summary history cleared.")

    # Show history
    if st.sidebar.checkbox("Show Summary History"):
        history = load_summary_history()
        if history:
            st.write("Summary History:")
            st.text_area("History", history, height=300)
        else:
            st.write("No history available.")

if __name__ == "__main__":
    main()
