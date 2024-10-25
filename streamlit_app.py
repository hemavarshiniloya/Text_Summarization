import streamlit as st
from transformers import pipeline

class TextSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        """Initialize the summarization pipeline with the specified model."""
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize(self, text):
        """Summarize the input text."""
        if len(text) < 50:
            return "Text is too short for summarization."
        summary = self.summarizer(text, max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']

def main():
    """Main function to run the Streamlit app."""
    st.title("Text Summarization App")

    # Sidebar for model selection
    st.sidebar.title("Model Selection")
    model_options = {
        "BART": "facebook/bart-large-cnn",
        "T5 Small": "t5-small",
        "T5 Base": "t5-base",
    }
    selected_model = st.sidebar.selectbox("Select a model for summarization", list(model_options.keys()))
    model_name = model_options[selected_model]

    # Initialize the summarizer
    summarizer = TextSummarizer(model_name)

    # Input text area
    input_text = st.text_area("Enter text to summarize:", height=300)

    if st.button("Summarize"):
        if input_text:
            with st.spinner("Summarizing..."):
                summary = summarizer.summarize(input_text)
                st.subheader("Summary:")
                st.write(summary)
        else:
            st.error("Please enter some text to summarize.")

if __name__ == "__main__":
    main()
