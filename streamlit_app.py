import streamlit as st
import nltk
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import PyPDF2
import docx

# Download necessary NLTK data if not already present
def download_nltk_data():
    """Download required NLTK data for TextBlob."""
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

# Extract text from a given URL
def extract_text_from_url(url):
    """Extract text from a given URL."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return ' '.join(p.get_text() for p in soup.find_all('p'))

# Extract text from PDF
def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Extract text from Word document
def extract_text_from_docx(file):
    """Extract text from a Word document."""
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

# Summarize the given text using TextBlob
def summarize_text(text, num_sentences=3):
    """Summarize the given text using TextBlob."""
    download_nltk_data()  # Ensure required NLTK data is downloaded
    blob = TextBlob(text)
    sentences = blob.sentences
    sorted_sentences = sorted(sentences, key=lambda s: len(s.noun_phrases), reverse=True)
    summary = ' '.join(str(sentence) for sentence in sorted_sentences[:num_sentences])
    return summary

# Main function for the Streamlit app
def main():
    st.title("Text Summarization App")

    choice = st.sidebar.radio("Choose an option", ["Summarize Text", "Summarize URL", "Summarize Document"])

    if choice == "Summarize Text":
        input_text = st.text_area("Enter your text here:", height=300)
        num_sentences = st.number_input("Number of sentences for summary:", min_value=1, max_value=10, value=3)

        if st.button("Summarize"):
            if input_text:
                summary = summarize_text(input_text, num_sentences)
                st.write("### Summary")
                st.write(summary)

    elif choice == "Summarize URL":
        url = st.text_input("Enter URL:")
        num_sentences = st.number_input("Number of sentences for summary:", min_value=1, max_value=10, value=3)

        if st.button("Summarize"):
            if url:
                text_from_url = extract_text_from_url(url)
                if text_from_url:
                    summary = summarize_text(text_from_url, num_sentences)
                    st.write("### Summary")
                    st.write(summary)
                else:
                    st.write("No text found at the provided URL.")

    elif choice == "Summarize Document":
        uploaded_file = st.file_uploader("Upload a PDF or Word document", type=["pdf", "docx"])
        num_sentences = st.number_input("Number of sentences for summary:", min_value=1, max_value=10, value=3)

        if st.button("Summarize"):
            if uploaded_file:
                if uploaded_file.type == "application/pdf":
                    text_from_document = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text_from_document = extract_text_from_docx(uploaded_file)

                if text_from_document:
                    summary = summarize_text(text_from_document, num_sentences)
                    st.write("### Summary")
                    st.write(summary)
                else:
                    st.write("No text found in the uploaded document.")

if __name__ == "__main__":
    main()
