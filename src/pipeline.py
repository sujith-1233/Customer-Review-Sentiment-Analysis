import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_demo_data():
    path = __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "sample_reviews.csv"
    return pd.read_csv(path)

def _normalise_labels(series):
    s = series.astype(str).str.strip().str.lower()
    mapping = {"1":"positive", "0":"negative", "pos":"positive", "neg":"negative",
               "positive":"positive", "negative":"negative"}
    s = s.map(lambda x: mapping.get(x, x))
    return s

def train_and_evaluate(df):
    if "review" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("Training requires 'review' and 'sentiment' columns.")
    data = df[["review", "sentiment"]].dropna()
    data["sentiment"] = _normalise_labels(data["sentiment"])
    data = data[data["sentiment"].isin(["positive", "negative"])]
    if len(data) < 10 or data["sentiment"].nunique() < 2:
        raise ValueError("Provide at least 10 labelled reviews and examples from both positive and negative classes.")
    if data["sentiment"].value_counts().min() < 2:
        raise ValueError("Each sentiment class needs at least two examples.")
    X = data["review"].astype(str).map(clean_text)
    y = data["sentiment"]
    test_count = max(2, int(round(len(data) * .2)))
    if test_count >= len(data):
        test_count = 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_count, random_state=42, stratify=y
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(class_weight="balanced")
    }
    bundle = {"models": {}, "evaluation": {}, "vectorizer": None, "word2vec_note": ""}
    metrics = {}
    for name, clf in models.items():
        pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=12000,
                                                   min_df=1, sublinear_tf=True)),
                         ("classifier", clf)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, pred, labels=["negative", "positive"])
        scores = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, pos_label="positive", zero_division=0),
            "recall": recall_score(y_test, pred, pos_label="positive", zero_division=0),
            "f1_score": f1_score(y_test, pred, pos_label="positive", zero_division=0)
        }
        metrics[name] = scores
        bundle["models"][name] = pipe
        bundle["evaluation"][name] = {"confusion_matrix": cm.tolist(), "report": classification_report(
            y_test, pred, labels=["negative","positive"], zero_division=0, output_dict=True)}
    # Train an independent Word2Vec representation and document it in the project.
    # It is optional at runtime because gensim may require platform-specific SciPy compatibility.
    try:
        from gensim.models import Word2Vec
        tokenized = [clean_text(t).split() for t in data["review"].astype(str)]
        w2v = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=1, epochs=30)
        bundle["word2vec_model"] = w2v
        bundle["word2vec_note"] = "Word2Vec trained on the supplied review corpus (100-dimensional embeddings)."
    except Exception as exc:
        bundle["word2vec_model"] = None
        bundle["word2vec_note"] = f"Word2Vec optional dependency unavailable: {exc}"
    bundle["best_model"] = max(metrics, key=lambda k: metrics[k]["f1_score"])
    return bundle, metrics

def predict_reviews(reviews, bundle):
    model = bundle["models"][bundle["best_model"]]
    cleaned = [clean_text(x) for x in reviews]
    preds = model.predict(cleaned)
    clf = model.named_steps["classifier"]
    if hasattr(clf, "predict_proba"):
        confidence = clf.predict_proba(model.named_steps["tfidf"].transform(cleaned)).max(axis=1)
    elif hasattr(clf, "decision_function"):
        d = np.abs(clf.decision_function(model.named_steps["tfidf"].transform(cleaned)))
        confidence = 1 - np.exp(-d)
    else:
        confidence = np.full(len(preds), np.nan)
    return pd.DataFrame({"review": reviews, "predicted_sentiment": preds, "confidence": confidence})

def summarize_reviews(df):
    if df is None or len(df) == 0:
        return "No reviews are currently loaded."
    if "sentiment" in df.columns:
        counts = _normalise_labels(df["sentiment"]).value_counts()
        total = int(counts.sum())
        pos, neg = int(counts.get("positive",0)), int(counts.get("negative",0))
        return f"Analyzed {total} labelled reviews: {pos} positive ({pos/total:.1%}) and {neg} negative ({neg/total:.1%})."
    return f"{len(df)} reviews loaded. Train a model to generate sentiment-based opinion summaries."
