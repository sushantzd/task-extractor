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

def extract_person(sentence):
    """Extract the responsible person using Named Entity Recognition (NER)."""
    doc = get_nlp()(sentence)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
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
