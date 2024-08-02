from transformers import pipeline

# Initialize text summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Initialize sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Function to perform summarization
def text_summary(text, maxlength=150):
    result = summarizer(text, max_length=maxlength, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    return result[0]['summary_text']

# Function to perform sentiment analysis
def sentiment_analysis(text):
    result = sentiment_analyzer(text)[0]
    return {"label": result['label'], "score": result['score']}
