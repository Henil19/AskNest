# AskNest

A beginner-friendly FAQ chatbot for headphone questions. It preprocesses the existing FAQ questions with spaCy, represents them with TF-IDF, and uses cosine similarity to select a response. Gradio provides a local chat interface.

## Project files

- `chatbot_for_faqs.py` — application and FAQ matching logic.
- `faqs.csv` — source questions and answers (keep it beside the Python file).
- `requirements.txt` — Python package dependencies.
- `.gitignore` — local environment and generated-file exclusions.

## Requirements

Use Python 3.9 or newer. Install the packages and the spaCy English model:

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run locally

From the project directory, run:

```bash
python chatbot_for_faqs.py
```

The app prints a local Gradio URL in the terminal. Open it in a browser and ask a headphone FAQ. The app also locates `faqs.csv` relative to its own file, so it can be started from another working directory.

## Matching behavior

The app lowercases FAQ and user questions, removes stop words and non-alphabetic tokens, and lemmatizes the remaining words with spaCy. It builds TF-IDF vectors from the FAQ questions and compares a user question with them using cosine similarity. At a similarity score of 25% or higher, it returns the corresponding answer and shows the matched FAQ and score. Below that threshold, it reports that no confident answer was found and shows the closest-match score without presenting an answer or matched FAQ.

The threshold is a simple starting point for this small sample dataset. Adjust `MATCH_THRESHOLD` in `chatbot_for_faqs.py` after trying representative questions if you expand the FAQs.
