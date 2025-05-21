# TaskExtractor-NLP

This project demonstrates a rule-based Natural Language Processing (NLP) pipeline to extract and categorize tasks (also known as action items) from unstructured or free-form text.

## 🎯 Objective

To identify actionable sentences (tasks) from a given paragraph and extract:
- The **task description**
- The **person/entity** responsible for it (if present)
- The **deadline/time** (if mentioned)
- And **categorize** the tasks based on similarity

> ⚠️ No pre-trained models or LLMs (like GPT, BERT classifiers) were used. The approach relies entirely on **heuristics**, **POS tagging**, and **rule-based logic**.

---

## 🛠️ Techniques & Tools Used

- Python
- NLTK / spaCy (for text processing, POS tagging, NER)
- Regex and heuristic rules (to identify tasks)
- Word2Vec / BERT (to generate word embeddings)
- KMeans / LDA (for task categorization)


## 🚀 How It Works

1. **Preprocessing**
   - Clean text (punctuation, stop words, etc.)
   - Sentence tokenization
   - POS tagging and NER

2. **Task Identification**
   - Heuristic rules such as:
     - Sentences with imperative verbs (e.g., "buy", "submit")
     - Named entities (e.g., "Rahul has to...")
     - Deadlines or time-related phrases ("by 5 pm", "tomorrow")

3. **Categorization**
   - Extract task embeddings using Word2Vec / BERT
   - Group tasks into similar categories using clustering (KMeans) or LDA

4. **Output**
   - A structured list with:
     - `task`
     - `assigned_to`
     - `deadline`
     - `category`

---

## 📌 Example Input & Output

**Input:**
> Rahul wakes up early every day. He goes to college in the morning and comes back at 3 pm. At present, Rahul is outside. He has to buy the snacks for all of us.

**Output:**
```json
{
  "task": "buy the snacks for all of us",
  "assigned_to": "Rahul",
  "deadline": null,
  "category": "Personal Errands"
}
