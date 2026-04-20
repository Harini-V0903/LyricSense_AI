import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/songs.csv")

# =========================
# HANDLE COLUMN VARIANTS
# =========================
# If dataset uses different naming, adjust automatically
if "lyrics" in df.columns:
    X = df["lyrics"]
elif "Lyrics" in df.columns:
    X = df["Lyrics"]
else:
    raise Exception("No lyrics column found")

if "mood" in df.columns:
    y = df["mood"]
elif "Mood" in df.columns:
    y = df["Mood"]
else:
    raise Exception("No mood column found")

df = pd.DataFrame({"Lyrics": X, "Mood": y}).dropna()

# =========================
# CLEAN TEXT
# =========================
def clean(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

df["Lyrics"] = df["Lyrics"].apply(clean)

X = df["Lyrics"]
y = df["Mood"]

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# MODEL PIPELINE
# =========================
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )),
    ("clf", LogisticRegression(max_iter=300))
])

# =========================
# TRAIN
# =========================
model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "model/song_mood_model.pkl")

print("\nModel saved successfully 🚀")