from fastapi import FastAPI
from pydantic import BaseModel
from src.model import TicketClassifier

app = FastAPI(title="Ticket Auto-Triage API")

class TicketIn(BaseModel):
    subject: str = ""
    description: str = ""

@app.on_event("startup")
def load_model():
    global clf
    clf = TicketClassifier(path="models/ticket_clf.joblib")

@app.post('/predict')
def predict(ticket: TicketIn):
    return clf.predict(ticket.subject, ticket.description)
