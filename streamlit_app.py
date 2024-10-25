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
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.]', '', text)
        return text
    
    def get_sentences(self, text):
        sentences = re.split(r'[.!?]+', text)
        return [sent.strip() for sent in sentences if sent.strip()]
    
    def get_word_frequency(self, text):
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
            cleaned_text = self.preprocess_text(text)
            sentences = self.get_sentences(cleaned_text)
            
            if not sentences:
                return "Could not generate summary. Text too short or invalid."
            
            num_sentences = min(num_sentences, len(sentences))
            word_freq = self.get_word_frequency(cleaned_text)
            sentence_scores = self.score_sentences(sentences, word_freq)
            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
            summary = '. '.join(sent for sent, score in top_sentences)
            return summary.capitalize() + '.'
            
        except Exception as e:
            return f"An error occurred while summarizing: {str(e)}"

def extract_text_from_url(url):
    try:
        headers = {
            'User -Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        st.error(f"Error extracting text from URL: {str(e)}")
        return None

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

def main():
    st.set_page_config(
        page_title="Text Summarizer",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Text Summarizer")
    st.write("Upload text, URL, or documents to generate a summary.")

    # Initialize summarizer
    summarizer = TextSummarizer()

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📝 Text", "🌐 URL", "📄 Document"])

    with tab1:
        st.header("📝 Text Input")
        text_input = st.text_area("Enter your text here:", height=200)
        num_sentences = st.slider("Number of sentences in summary:", min_value=1, max_value=10, value=3)
        
        if st.button("Summarize Text", key="text_button"):
            if text_input:
                with st.spinner("Generating summary... "):
                    summary = summarizer.summarize(text_input, num_sentences)
                    st.success("Summary generated!")
                    st.write(summary)
            else:
                st.warning("Please enter some text to summarize.")

    with tab2:
        st.header("🌐 URL Input")
        url_input = st.text_input("Enter URL:")
        url_sentences = st.slider("Number of sentences in summary:", min_value=1, max_value=10, value=3, key="url_slider")
        
        if st.button("Summarize URL", key="url_button"):
            if url_input:
                with st.spinner("Fetching content and generating summary..."):
                    text = extract_text_from_url(url_input)
                    if text:
                        summary = summarizer.summarize(text, url_sentences)
                        st.success("Summary generated!")
                        st.write(summary)
                    else:
                        st.error("Could not extract text from the URL.")
            else:
                st.warning("Please enter a URL.")

    with tab3:
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader("Upload a document", type=['pdf', 'docx'])
        doc_sentences = st.slider("Number of sentences in summary:", min_value=1, max_value=10, value=3, key="doc_slider")
        
        if uploaded_file and st.button("Summarize Document", key="doc_button"):
            with st.spinner("Processing document and generating summary..."):
                file_type = uploaded_file.type
                if file_type == "application/pdf":
                    text = extract_text_from_pdf(uploaded_file)
                elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text = extract_text_from_docx(uploaded_file)
                else:
                    text = None
                    st.error("Unsupported file type")

                if text:
                    summary = summarizer.summarize(text, doc_sentences)
                    st.success("Summary generated!")
                    st.write(summary)

    # Add footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center">
            <p>Text Summarization App</p>
            <i class="fa fa-copyright" aria-hidden="true"></i> 2023
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
