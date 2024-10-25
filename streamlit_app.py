import streamlit as st
import requests
from bs4 import BeautifulSoup
import PyPDF2
import docx
from collections import Counter
import re

class TextSummarizer:
    def __init__(self):
        pass
        
    def preprocess_text(self, text):
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.]', '', text)
        return text
    
    def get_sentences(self, text):
        # Simple sentence tokenization
        sentences = re.split(r'[.!?]+', text)
        return [sent.strip() for sent in sentences if sent.strip()]
    
    def get_word_frequency(self, text):
        # Remove common words and get word frequency
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = text.split()
        word_freq = Counter()
        
        for word in words:
            if word not in common_words:
                word_freq[word] += 1
                
        return word_freq
    
    def score_sentences(self, sentences, word_freq):
        sentence_scores = {}
        for sentence in sentences:
            words = sentence.split()
            score = sum(word_freq.get(word, 0) for word in words)
            sentence_scores[sentence] = score
        return sentence_scores
    
    def summarize(self, text, num_sentences=3):
        try:
            # Preprocess text
            cleaned_text = self.preprocess_text(text)
            
            # Get sentences
            sentences = self.get_sentences(cleaned_text)
            
            if not sentences:
                return "Could not generate summary. Text too short or invalid."
            
            # Limit num_sentences to available sentences
            num_sentences = min(num_sentences, len(sentences))
            
            # Get word frequency
            word_freq = self.get_word_frequency(cleaned_text)
            
            # Score sentences
            sentence_scores = self.score_sentences(sentences, word_freq)
            
            # Get top sentences
            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
            
            # Reconstruct summary
            summary = '. '.join(sent for sent, score in top_sentences)
            
            return summary.capitalize() + '.'
            
        except Exception as e:
            return f"An error occurred while summarizing: {str(e)}"

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def extract_text_from_docx(docx_file):
    try:
        doc = docx.Document(docx_file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {str(e)}")
        return None

def extract_text_from_txt(txt_file):
    try:
        text = txt_file.read().decode("utf-8")
        return text
    except Exception as e:
        st.error(f"Error extracting text from TXT: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Multi-Document Summarizer",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Multi-Document Summarizer")
    st.write("Upload multiple documents (PDF, DOCX, TXT) to generate summaries.")

    # Initialize summarizer
    summarizer = TextSummarizer()

    # Create a file uploader for multiple files
    uploaded_files = st.file_uploader("Upload documents (PDF, DOCX, TXT)", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
    
    num_sentences = st.slider("Number of sentences in summary:", min_value=1, max_value=10, value=3)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.subheader(f"Processing {uploaded_file.name}...")
            text = ""
            
            # Extract text based on file type
            if uploaded_file.type == "application/pdf":
                text = extract_text_from_pdf(upload _file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = extract_text_from_docx(upload_file)
            elif uploaded_file.type == "text/plain":
                text = extract_text_from_txt(upload_file)
            
            if text:
                summary = summarizer.summarize(text, num_sentences)
                st.write(f"Summary of {uploaded_file.name}:")
                st.write(summary)
            else:
                st.write(f"Failed to extract text from {uploaded_file.name}. Skipping.")

if __name__ == "__main__":
    main()
