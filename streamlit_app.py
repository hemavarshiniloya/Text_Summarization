import streamlit as st
import nltk
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import PyPDF2
from docx import Document
import subprocess
import sys
import os

# Function to ensure NLTK data is downloaded
def download_nltk_data():
    """Download required NLTK data."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')

# Function to download TextBlob corpora
def download_textblob_data():
    """Download TextBlob corpora."""
    try:
        subprocess.check_call([
            sys.executable,
            "-m",
            "textblob.download_corpora"
        ])
    except subprocess.CalledProcessError:
        st.error("Failed to download TextBlob corpora")
        raise

# Initialize required data
@st.cache_resource
def initialize_nlp_resources():
    """Initialize all NLP resources."""
    download_nltk_data()
    download_textblob_data()
    return True

# Text summarization function
def summarize_text(text, num_sentences=3):
    """Generate text summary using TextBlob."""
    try:
        # Ensure NLP resources are initialized
        if not hasattr(summarize_text, "initialized"):
            initialize_nlp_resources()
            summarize_text.initialized = True

        # Create TextBlob object
        blob = TextBlob(text)
        
        # Get sentences
        sentences = blob.sentences
        
        if not sentences:
            return "No sentences found in the text."
        
        # Score sentences based on the number of noun phrases
        sentence_scores = []
        for sentence in sentences:
            score = len(sentence.noun_phrases)
            sentence_scores.append((sentence, score))
        
        # Sort sentences by score
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top sentences
        top_sentences = sentence_scores[:num_sentences]
        
        # Sort sentences by their original order
        top_sentences.sort(key=lambda x: text.find(str(x[0])))
        
        # Join sentences into final summary
        summary = ' '.join(str(sentence[0]) for sentence in top_sentences)
        return summary

    except Exception as e:
        st.error(f"Error during summarization: {str(e)}")
        return ""

# Function to extract text from URL
def extract_text_from_url(url):
    """Extract text from a webpage."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join([p.get_text() for p in paragraphs])
    except Exception as e:
        st.error(f"Error extracting text from URL: {str(e)}")
        return ""

# Function to extract text from PDF
def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text() + '\n'
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

# Function to extract text from DOCX
def extract_text_from_docx(file):
    """Extract text from a Word document."""
    try:
        doc = Document(file)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {str(e)}")
        return ""

def main():
    st.title("Text Summarization App")
    
    # Initialize NLP resources at startup
    with st.spinner("Initializing NLP resources..."):
        try:
            initialize_nlp_resources()
            st.success("NLP resources initialized successfully!")
        except Exception as e:
            st.error(f"Failed to initialize NLP resources: {str(e)}")
            return

    # Input selection
    input_method = st.radio(
        "Choose input method:",
        ["Text Input", "URL", "Upload Document"]
    )

    # Text input
    if input_method == "Text Input":
        text = st.text_area("Enter text to summarize:", height=200)
        num_sentences = st.slider("Number of sentences in summary:", 1, 10, 3)
        
        if st.button("Summarize"):
            if text:
                with st.spinner("Generating summary..."):
                    summary = summarize_text(text, num_sentences)
                    if summary:
                        st.subheader("Summary:")
                        st.write(summary)
                        # Add download button
                        st.download_button(
                            label="Download Summary",
                            data=summary,
                            file_name="summary.txt",
                            mime="text/plain"
                        )
            else:
                st.warning("Please enter some text to summarize.")

    # URL input
    elif input_method == "URL":
        url = st.text_input("Enter URL:")
        num_sentences = st.slider("Number of sentences in summary:", 1, 10, 3)
        
        if st.button("Summarize"):
            if url:
                with st.spinner("Fetching and summarizing content..."):
                    text = extract_text_from_url(url)
                    if text:
                        summary = summarize_text(text, num_sentences)
                        if summary:
                            st.subheader("Summary:")
                            st.write(summary)
                            st.download_button(
                                label="Download Summary",
                                data=summary,
                                file_name="summary.txt",
                                mime="text/plain"
                            )
                    else:
                        st.error("Could not extract text from URL.")
            else:
                st.warning("Please enter a URL.")

    # Document upload
    else:
        uploaded_file = st.file_uploader("Upload document", type=["pdf", "docx"])
        num_sentences = st.slider("Number of sentences in summary:", 1, 10, 3)
        
        if uploaded_file is not None:
            if st.button("Summarize"):
                with st.spinner("Processing document..."):
                    if uploaded_file.type == "application/pdf":
                        text = extract_text_from_pdf(uploaded_file)
                    else:
                        text = extract_text_from_docx(uploaded_file)
                        
                    if text:
                        summary = summarize_text(text, num_sentences)
                        if summary:
                            st.subheader("Summary:")
                            st.write(summary)
                            st.download_button(
                                label="Download Summary",
                                data=summary,
                                file_name="summary.txt",
                                mime="text/plain"
                            )
                    else:
                        st.error("Could not extract text from document.")

if __name__ == "__main__":
    main()
