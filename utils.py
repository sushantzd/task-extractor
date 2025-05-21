import logging
from preprocessing import clean_text
from task_identification import extract_tasks
from categorization import categorize_tasks

def process_text(raw_text):
    """
    Process the raw input text through the NLP pipeline:
    - Extract tasks from text.
    - Categorize tasks and extract topics.
    """
    try:
        if not raw_text.strip():
            logging.warning("Empty input text received.")
            return [], []
        
        tasks = extract_tasks(raw_text)
        
        if not tasks:
            logging.info("No actionable tasks detected.")
            return [], []
        
        categorized_tasks, topics = categorize_tasks(tasks)
        
        return categorized_tasks, topics
    except Exception as e:
        logging.error(f"Error processing text: {e}")
        return [], []

def read_file(uploaded_file):
    """
    Reads an uploaded file and returns its content as a string.
    """
    try:
        file_bytes = uploaded_file.read()
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        return str(e)
