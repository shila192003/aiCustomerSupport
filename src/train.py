import argparse
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd
import numpy as np
from src.utils import load_csv, join_text
from sklearn.metrics import classification_report, accuracy_score


def build_pipeline():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=20000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    return pipe


def main(args):
    print("Loading data from", args.data)
    df = load_csv(args.data)
    df = df.dropna(subset=["Category"]) 
    texts = df.apply(lambda r: join_text(r.get("Subject", ""), r.get("Description", "")), axis=1)
    y = df["Category"].astype(str)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Only stratify if every class has at least 2 samples (required by stratify)
    try:
        unique, counts = np.unique(y_enc, return_counts=True)
    except Exception:
        unique, counts = np.unique(y_enc), np.array([1])
    if counts.min() < 2:
        print("Warning: some classes have fewer than 2 samples — disabling stratify for train_test_split")
        strat = None
    else:
        strat = y_enc

    X_train, X_test, y_train, y_test = train_test_split(
        texts, y_enc, test_size=0.2, random_state=42, stratify=strat
    )

    pipe = build_pipeline()
    print("Fitting pipeline on", len(X_train), "examples")
    try:
        pipe.fit(X_train, y_train)
    except Exception as e:
        print("Error during pipeline.fit:", e)
        raise

    preds = pipe.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, preds))
    # classification_report requires target_names to match number of labels present
    present_labels = np.unique(y_test)
    try:
        if len(present_labels) != len(le.classes_):
            target_names = le.inverse_transform(present_labels)
            print("Note: test set contains a subset of classes; showing report for present classes only.")
            print(classification_report(y_test, preds, labels=present_labels, target_names=target_names))
        else:
            print(classification_report(y_test, preds, target_names=le.classes_))
    except Exception as e:
        print("Warning: could not print full classification_report:", e)
        print(classification_report(y_test, preds))

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("Saving model to", args.out)
    try:
        joblib.dump({"pipeline": pipe, "label_encoder": le}, args.out)
    except Exception as e:
        print("Error saving model:", e)
        raise
    print("Saved model to", args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV file")
    parser.add_argument("--out", default="models/ticket_clf.joblib", help="Output model path")
    args = parser.parse_args()
    main(args)
