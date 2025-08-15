import re
import nltk
import spacy
from preprocessing import tokenize_sentences, clean_text

_nlp = None

def get_nlp():
    """Load spaCy model lazily and download if missing."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


TASK_KEYWORDS = ["has to", "should", "must", "needs to", "is required to"]

import re
import nltk
from nltk import pos_tag, word_tokenize

def extract_person(sentence):
    doc = get_nlp()(sentence)
    ignore_words = {"purchase", "buy", "shop", "order", "clean", "finish", "complete"}

    # 1. Try spaCy NER first
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            if not any(word.lower() in ignore_words for word in ent.text.split()):
                return ent.text

    # 2. Fallback: check for proper nouns (NNP) with NLTK
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    for word, tag in tagged:
        if tag == "NNP" and word.lower() not in ignore_words:
            return word

    return "Not Specified"


def extract_deadline(sentence):
    """Extract deadline information using regex."""
    deadline_pattern = r'\b(?:by|before|at)\s+((?:\d{1,2}\s*(?:am|pm))|tomorrow|today|next\s+\w+day|end of day)'
    match = re.search(deadline_pattern, sentence, re.IGNORECASE)
    return match.group(1) if match else "Not Specified"

def extract_tasks(raw_text):
    """Process raw input text to extract actionable tasks."""
    sentences = tokenize_sentences(raw_text)
    tasks = []
    
    for sentence in sentences:
        cleaned_sentence = clean_text(sentence)
        if any(keyword in cleaned_sentence for keyword in TASK_KEYWORDS):
            person = extract_person(sentence)
            deadline = extract_deadline(cleaned_sentence)
            tasks.append({
                'task': sentence,
                'person': person,
                'deadline': deadline
            })
    return tasks
