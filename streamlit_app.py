import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import requests
from bs4 import BeautifulSoup
import PyPDF2
from docx import Document
import re

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK data: {str(e)}")
        return False

# Text preprocessing
def preprocess_text(text):
    """Clean and preprocess the text."""
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    
    # Remove short sentences
    sentences = [sentence for sentence in sentences if len(sentence.split()) > 3]
    
    return sentences

# Method 1: TextRank-based summarization
def textrank_summarize(text, num_sentences=3):
    """Implement TextRank algorithm for text summarization."""
    sentences = preprocess_text(text)
    
    # Create similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 != idx2:
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2])
    
    # Create graph and apply PageRank
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    
    # Sort sentences by score
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    # Select top sentences
    summary = ' '.join([ranked_sentences[i][1] for i in range(min(num_sentences, len(ranked_sentences)))])
    return summary

# Method 2: TF-IDF based summarization
def tfidf_summarize(text, num_sentences=3):
    """Implement TF-IDF based summarization."""
    sentences = preprocess_text(text)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Calculate sentence scores
    sentence_scores = []
    for i in range(len(sentences)):
        score = np.sum(tfidf_matrix[i].toarray())
        sentence_scores.append((score, sentences[i]))
    
    # Sort and select top sentences
    ranked_sentences = sorted(sentence_scores, reverse=True)
    summary = ' '.join([ranked_sentences[i][1] for i in range(min(num_sentences, len(ranked_sentences)))])
    return summary

# Method 3: Frequency-based summarization
def frequency_summarize(text, num_sentences=3):
    """Implement frequency-based summarization."""
    sentences = preprocess_text(text)
    
    # Tokenize words and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word not in stop_words and word.isalnum()]
    
    # Calculate word frequencies
    freq = Counter(words)
    
    # Score sentences based on word frequencies
    sentence_scores = []
    for sentence in sentences:
        score = sum(freq[word.lower()] for word in word_tokenize(sentence) 
                   if word.lower() in freq)
        sentence_scores.append((score, sentence))
    
    # Sort and select top sentences
    ranked_sentences = sorted(sentence_scores, reverse=True)
    summary = ' '.join([ranked_sentences[i][1] for i in range(min(num_sentences, len(ranked_sentences)))])
    return summary

# Helper function for TextRank
def sentence_similarity(sent1, sent2):
    """Calculate similarity between two sentences."""
    words1 = set(word_tokenize(sent1))
    words2 = set(word_tokenize(sent2))
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words1 = words1.difference(stop_words)
    words2 = words2.difference(stop_words)
    
    # Calculate Jaccard similarity
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if len(union) > 0 else 0

# Text extraction functions
def extract_text_from_url(url):
    """Extract text from URL."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join([p.get_text() for p in paragraphs])
    except Exception as e:
        st.error(f"Error extracting text from URL: {str(e)}")
        return ""

def extract_text_from_pdf(file):
    """Extract text from PDF."""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text() + '\n'
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def extract_text_from_docx(file):
    """Extract text from DOCX."""
    try:
        doc = Document(file)
        return ' '.join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {str(e)}")
        return ""

# Main Streamlit app
def main():
    st.title("Advanced Text Summarization App")
    
    # Initialize NLTK data
    if not download_nltk_data():
        st.error("Failed to initialize required data. Please try again.")
        return

    # Sidebar options
    st.sidebar.title("Settings")
    summarization_method = st.sidebar.selectbox(
        "Choose Summarization Method",
        ["TextRank", "TF-IDF", "Frequency-Based"]
    )
    
    input_method = st.sidebar.radio(
        "Choose Input Method",
        ["Text Input", "URL", "Document Upload"]
    )

    # Main content
    if input_method == "Text Input":
        text = st.text_area("Enter your text here:", height=200)
        num_sentences = st.slider("Number of sentences in summary:", 1, 10, 3)
        
        if st.button("Generate Summary"):
            if text:
                with st.spinner("Generating summary..."):
                    if summarization_method == "TextRank":
                        summary = textrank_summarize(text, num_sentences)
                    elif summarization_method == "TF-IDF":
                        summary = tfidf_summarize(text, num_sentences)
                    else:
                        summary = frequency_summarize(text, num_sentences)
                    
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

    elif input_method == "URL":
        url = st.text_input("Enter URL:")
        num_sentences = st.slider("Number of sentences in summary:", 1, 10, 3)
        
        if st.button("Generate Summary"):
            if url:
                with st.spinner("Fetching and summarizing content..."):
                    text = extract_text_from_url(url)
                    if text:
                        if summarization_method == "TextRank":
                            summary = textrank_summarize(text, num_sentences)
                        elif summarization_method == "TF-IDF":
                            summary = tfidf_summarize(text, num_sentences)
                        else:
                            summary = frequency_summarize(text, num_sentences)
                        
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
                        st.error("Could not extract text from URL.")
            else:
                st.warning("Please enter a URL.")

    else:
        uploaded_file = st.file_uploader("Upload document", type=["pdf", "docx"])
        num_sentences = st.slider("Number of sentences in summary:", 1, 10, 3)
        
        if uploaded_file is not None:
            if st.button("Generate Summary"):
                with st.spinner("Processing document..."):
                    if uploaded_file.type == "application/pdf":
                        text = extract_text_from_pdf(uploaded_file)
                    else:
                        text = extract_text_from_docx(uploaded_file)
                        
                    if text:
                        if summarization_method == "TextRank":
                            summary = textrank_summarize(text, num_sentences)
                        elif summarization_method == "TF-IDF":
                            summary = tfidf_summarize(text, num_sentences)
                        else:
                            summary = frequency_summarize(text, num_sentences)
                        
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
                        st.error("Could not extract text from document.")

if __name__ == "__main__":
    main()
