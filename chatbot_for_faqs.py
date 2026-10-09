"""A small FAQ chatbot using spaCy, TF-IDF, cosine similarity, and Gradio."""

from pathlib import Path
import re

import gradio as gr
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
FAQS_PATH = BASE_DIR / "faqs.csv"
SPACY_MODEL = "en_core_web_sm"
MATCH_THRESHOLD = 0.25


def load_nlp():
    """Load spaCy's English model with a helpful setup error."""
    try:
        return spacy.load(SPACY_MODEL, disable=["ner", "parser"])
    except OSError as exc:
        raise RuntimeError(
            f"The spaCy model '{SPACY_MODEL}' is not installed. "
            "Install it with: python -m spacy download en_core_web_sm"
        ) from exc


def load_faqs(path=FAQS_PATH):
    """Read and validate the FAQ CSV."""
    if not path.is_file():
        raise FileNotFoundError(
            f"FAQ dataset not found: {path}. Place faqs.csv beside chatbot_for_faqs.py."
        )

    try:
        faqs = pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read FAQ dataset '{path}': {exc}") from exc

    # Accept CSV headers regardless of capitalization or surrounding spaces.
    faqs.columns = [str(column).strip().lower() for column in faqs.columns]
    missing = {"question", "answer"} - set(faqs.columns)
    if missing:
        raise ValueError(
            "FAQ dataset must contain 'Question' and 'Answer' columns; "
            f"missing: {', '.join(sorted(missing))}."
        )

    faqs = faqs[["question", "answer"]].dropna()
    faqs["question"] = faqs["question"].astype(str).str.strip()
    faqs["answer"] = faqs["answer"].astype(str).str.strip()
    # The source CSV numbers each question for display; remove only that prefix.
    faqs["question"] = faqs["question"].str.replace(r"^\s*\d+\.\s*", "", regex=True)
    faqs = faqs[(faqs["question"] != "") & (faqs["answer"] != "")].reset_index(drop=True)
    if faqs.empty:
        raise ValueError("FAQ dataset contains no usable question and answer rows.")
    return faqs


def preprocess_text(text, nlp):
    """Lowercase, remove stop words and non-letters, and lemmatize with spaCy."""
    doc = nlp(str(text).lower())
    words = [
        token.lemma_.strip()
        for token in doc
        if token.is_alpha and not token.is_stop and len(token.lemma_.strip()) >= 2
    ]
    return " ".join(words)


def build_matcher(faqs, nlp):
    """Prepare processed FAQ questions and their TF-IDF matrix."""
    processed_questions = [preprocess_text(question, nlp) for question in faqs["question"]]
    if not any(processed_questions):
        raise ValueError("None of the FAQ questions contain searchable words.")

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(processed_questions)
    return vectorizer, matrix


def get_best_faq_answer(user_question, faqs, nlp, vectorizer, matrix,
                        threshold=MATCH_THRESHOLD):
    """Return the best answer only when its similarity clears the threshold."""
    user_question = str(user_question).strip()
    if not user_question:
        return {"answer": "Please enter a question.", "matched_question": None, "score": 0.0}

    processed = preprocess_text(user_question, nlp)
    if not processed:
        return {
            "answer": "I couldn't find a confident answer. Please try rephrasing your question.",
            "matched_question": None,
            "score": 0.0,
        }

    similarities = cosine_similarity(vectorizer.transform([processed]), matrix).ravel()
    best_index = int(similarities.argmax())
    score = float(similarities[best_index])
    if score >= threshold:
        answer = faqs.loc[best_index, "answer"]
    else:
        answer = "I couldn't find a confident answer in the FAQ. Please try rephrasing or ask about another topic."

    return {
        "answer": answer,
        "matched_question": (
            faqs.loc[best_index, "question"] if score >= threshold else None
        ),
        "score": score,
        "confident": score >= threshold,
    }


def create_chatbot():
    """Load data and matching components, then create the Gradio interface."""
    nlp = load_nlp()
    faqs = load_faqs()
    vectorizer, matrix = build_matcher(faqs, nlp)

    def respond(message, history):
        result = get_best_faq_answer(message, faqs, nlp, vectorizer, matrix)
        if result["matched_question"] is None:
            if "confident" in result:
                return (
                    f"{result['answer']}\n\n"
                    f"**Closest match score: {result['score']:.0%}**"
                )
            return result["answer"]
        confidence_label = "Similarity" if result.get("confident") else "Closest match score"
        return (
            f"{result['answer']}\n\n"
            f"**{confidence_label}: {result['score']:.0%}**\n"
            f"**Matched FAQ:** {result['matched_question']}"
        )

    return gr.ChatInterface(
        fn=respond,
        title="AskNest",
        description="Ask a question about the headphones in this FAQ dataset.",
        type="messages",
        textbox=gr.Textbox(placeholder="Type your question and press Enter...", label="Question"),
        examples=["How long does the battery last?", "How do I pair the headphones?", "How do I reset the headphones?"],
    )


def main():
    """Start the local web app."""
    try:
        app = create_chatbot()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Unable to start the FAQ chatbot: {exc}") from exc
    app.launch()


if __name__ == "__main__":
    main()

