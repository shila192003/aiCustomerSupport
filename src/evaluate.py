import argparse
import joblib
import pandas as pd
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from src.utils import load_csv, join_text

# Weights: accuracy 40%, precision&recall 30%, f1 20%, latency 10%
WEIGHTS = {"accuracy":0.4, "prec_recall":0.3, "f1":0.2, "latency":0.1}


def composite_score(acc, prec, rec, f1, latency_ms):
    # normalize latency to a 0-1 where lower latency is better
    # assume target latency 100ms -> score 1.0, 2000ms -> score 0.0 (clipped)
    lat = max(0.0, min(1.0, (2000.0 - latency_ms) / 1900.0))
    prec_rec = (prec + rec) / 2.0
    score = (WEIGHTS["accuracy"] * acc
             + WEIGHTS["prec_recall"] * prec_rec
             + WEIGHTS["f1"] * f1
             + WEIGHTS["latency"] * lat)
    return score


def main(args):
    data = load_csv(args.data)
    data = data.dropna(subset=["Category"]).reset_index(drop=True)
    texts = data.apply(lambda r: join_text(r.get("Subject", ""), r.get("Description", "")), axis=1)
    y_true = data["Category"].astype(str)

    d = joblib.load(args.model)
    pipe = d["pipeline"]
    le = d["label_encoder"]

    y_true_enc = le.transform(y_true)

    # measure latency over all examples
    start = time.time()
    probs = pipe.predict_proba(texts.tolist())
    preds = pipe.predict(texts.tolist())
    elapsed = (time.time() - start)
    avg_latency_ms = (elapsed / len(texts)) * 1000.0

    acc = accuracy_score(y_true_enc, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_enc, preds, average='macro')
    f1_macro = f1_score(y_true_enc, preds, average='macro')

    comp = composite_score(acc, precision, recall, f1_macro, avg_latency_ms)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1 (macro): {f1_macro:.4f}")
    print(f"Avg latency (ms): {avg_latency_ms:.2f}")
    print(f"Composite score: {comp:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='CSV with ground truth')
    parser.add_argument('--model', default='models/ticket_clf.joblib')
    args = parser.parse_args()
    main(args)
