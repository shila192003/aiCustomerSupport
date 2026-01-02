import pandas as pd
import numpy as np
import re


def load_csv(path):
    return pd.read_csv(path)


def normalize_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def join_text(subject: str, description: str) -> str:
    return " ".join([normalize_text(subject), normalize_text(description)])
