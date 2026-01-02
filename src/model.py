import joblib
import time

class TicketClassifier:
    def __init__(self, path="models/ticket_clf.joblib"):
        d = joblib.load(path)
        self.pipeline = d["pipeline"]
        self.le = d["label_encoder"]

    def predict(self, subject: str, description: str):
        text = (subject or "") + " " + (description or "")
        start = time.time()
        probs = self.pipeline.predict_proba([text])[0]
        pred_idx = int(self.pipeline.predict([text])[0])
        latency_ms = (time.time() - start) * 1000.0
        return {
            "category": self.le.inverse_transform([pred_idx])[0],
            "probabilities": {str(c): float(p) for c, p in zip(self.le.classes_, probs)},
            "latency_ms": latency_ms
        }
