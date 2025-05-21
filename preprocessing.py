import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

def clean_text(text):
    """
    Clean the input text by removing punctuation and converting it to lowercase.
    """
    text = re.sub(r"[^a-zA-Z0-9\s']", '', text)
    return text.lower()

def tokenize_sentences(text):
    """
    Tokenize the text into sentences.
    """
    return sent_tokenize(text)

def tokenize_words(sentence):
    """
    Tokenize a sentence into words.
    """
    return word_tokenize(sentence)

def remove_stopwords(tokens):
    """
    Remove stopwords while retaining important words.
    """
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens

def pos_tagging(tokens):
    """
    Perform POS tagging.
    """
    return nltk.pos_tag(tokens)
