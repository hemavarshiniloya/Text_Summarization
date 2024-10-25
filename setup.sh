#!/bin/bash

# Download TextBlob corpora
python -m textblob.download_corpora

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('brown'); nltk.download('wordnet'); nltk.download('omw-1.4')"
