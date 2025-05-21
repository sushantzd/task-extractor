import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim import corpora, models
import nltk

# Ensure required nltk packages are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Pre-defined categories with associated keywords
CATEGORY_KEYWORDS = {
    "Shopping": ["buy", "purchase", "shop", "order", "get"],
    "Cleaning": ["clean", "tidy", "wash", "sweep", "scrub"],
    "Communication": ["send", "email", "call", "talk", "discuss", "schedule"],
    "Review": ["review", "check", "verify", "inspect"],
    "Work": ["complete", "finish", "prepare", "submit", "work"],
    "Errand": ["pick", "drop", "collect", "deliver"]
}

def keyword_categorization(task_sentence):
    """
    Categorize a single task sentence using keyword matching.
    Returns the category with the highest keyword match, or "General" if no match is found.
    """
    tokens = [word.lower() for word in word_tokenize(task_sentence)]
    category_counts = {category: sum(token in keywords for token in tokens) for category, keywords in CATEGORY_KEYWORDS.items()}
    
    return max(category_counts, key=category_counts.get) if any(category_counts.values()) else "General"

def lda_categorization(task_sentences, num_topics=None):
    """
    Apply LDA topic modeling to a list of task sentences.
    """
    stop_words = set(stopwords.words('english'))
    processed_tasks = [[word.lower() for word in word_tokenize(sentence) if word.isalpha() and word.lower() not in stop_words] for sentence in task_sentences]
    
    dictionary = corpora.Dictionary(processed_tasks)
    corpus = [dictionary.doc2bow(text) for text in processed_tasks]
    
    num_topics = num_topics or min(3, len(dictionary))
    
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=42)
    
    topics = {f"Topic {idx+1}": [word for word, weight in lda_model.show_topic(idx, topn=5)] for idx in range(num_topics)}
    
    return topics

def categorize_tasks(tasks):
    """
    Assign a category to each task and generate LDA topics.
    """
    task_sentences = []
    for task in tasks:
        sentence = task.get('task', '')
        task['category'] = keyword_categorization(sentence)
        task_sentences.append(sentence)
    
    topics = lda_categorization(task_sentences) if task_sentences else {}

    return tasks, topics
