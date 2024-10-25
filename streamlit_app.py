import streamlit as st
import nltk
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import PyPDF2
from docx import Document
import io
import subprocess
import sys

# Initialize NLTK and TextBlob data
@st.cache_resource
def initialize_nlp():
    """Download required NLTK and TextBlob data."""
    try:
        # Download NLTK data
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        
        # Download TextBlob corpora
        subprocess.check_call([
            sys.executable, 
            "-m", 
            "textblob.download_corpora"
        ])
        return True
    except Exception as e:
        st.error(f"Error initializing NLP resources: {str(e)}")
        return False

# Text summarization function
def summarize_text(text, num_sentences=3):
    """Summarize the given text."""
    try:
        # Create TextBlob object
        blob = TextBlob(text)
        
        # Get sentences
        sentences = blob.sentences
        
        if not sentences:
            return "No sentences found in the text."
        
        # Calculate sentence scores based on noun phrases
        sentence_scores = []
        for sentence in sentences:
            # Get noun phrases
            noun_phrases = sentence.noun_phrases
            score = len(noun_phrases)
            sentence_scores.append((sentence, score))
        
        # Sort sentences by score
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top sentences
        top_sentences = [s[0] for s in sentence_scores[:num_sentences]]
        
        # Sort sentences by their original position
        top_sentences.sort(key=lambda s: sentences.index(s))
        
        # Join sentences
        summary = ' '.join(str(sentence) for sentence in top_sentences)
        return summary
    
    except Exception as e:
        st.error(f"Summarization error: {str(e)}")
        return ""

# URL text extraction
def extract_text_from_url(url):
    """Extract text from URL."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join([p.get_text() for p in paragraphs])
    except Exception as e:
        st.error(f"URL extraction error: {str(e)}")
        return ""

# PDF text extraction
def extract_text_from_pdf(file):
    """Extract text from PDF."""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text() + '\n'
        return text
    except Exception as e:
        st.error(f"PDF extraction error: {str(e)}")
        return ""

# DOCX text extraction
def extract_text_from_docx(file):
    """Extract text from DOCX."""
    try:
        doc = Document(file)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"DOCX extraction error: {str(e)}")
        return ""

# Save summary
def save_summary(summary, filename="summary.txt"):
    """Save summary to file."""
    try:
        btn = st.download_button(
            label="Download Summary",
            data=summary,
            file_name=filename,
            mime="text/plain"
        )
    except Exception as e:
        st.error(f"Save error: {str(e)}")

def main():
    st.title("Text Summarization App")
    
    # Initialize NLP resources
    with st.spinner("Initializing NLP resources..."):
        if not initialize_nlp():
            st.error("Failed to initialize NLP resources. Please try reloading the app.")
            return
    
    # Input method selection
    input_method = st.radio("Choose input method:", 
                           ["Text Input", "URL", "Upload Document"])
    
    if input_method == "Text Input":
        text_input = st.text_area("Enter text to summarize:", height=200)
        num_sentences = st.slider("Number of sentences in summary:", 1, 10, 3)
        
        if st.button("Summarize"):
            if text_input:
                with st.spinner("Generating summary..."):
                    summary = summarize_text(text_input, num_sentences)
                    if summary:
                        st.subheader("Summary")
                        st.write(summary)
                        save_summary(summary)
            else:
                st.warning("Please enter some text to summarize.")
                
    elif input_method == "URL":
        url_input = st.text_input("Enter URL:")
        num_sentences = st.slider("Number of sentences in summary:", 1, 10, 3)
        
        if st.button("Summarize"):
            if url_input:
                with st.spinner("Fetching and summarizing content..."):
                    text = extract_text_from_url(url_input)
                    if text:
                        summary = summarize_text(text, num_sentences)
                        if summary:
                            st.subheader("Summary")
                            st.write(summary)
                            save_summary(summary)
                    else:
                        st.error("Could not extract text from URL.")
            else:
                st.warning("Please enter a URL.")
                
    else:  # Upload Document
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
                            st.subheader("Summary")
                            st.write(summary)
                            save_summary(summary)
                    else:
                        st.error("Could not extract text from document.")

if __name__ == "__main__":
    main()
