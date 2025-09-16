import os, glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

DATASET_DIR = "dataset"
MODEL_OUT = "news_topic_pipeline.joblib"

texts, labels = [], []
for path in glob.glob(os.path.join(DATASET_DIR, "*.txt")):
    label = os.path.splitext(os.path.basename(path))[0]
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
                labels.append(label)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

clf = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
    ("logreg", LogisticRegression(max_iter=200, class_weight="balanced"))
])

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
joblib.dump(clf, MODEL_OUT)
print("Model saved to", MODEL_OUT)
