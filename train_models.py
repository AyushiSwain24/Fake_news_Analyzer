import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- MODEL 1: FAKE vs REAL (Upgraded) ---
print("\n--- Training Fake/Real Model (Logistic Regression + Bi-grams) ---")
df = pd.read_csv("fakereal_training_data.csv").dropna()
X = df['text']
y = df['label']

# ngram_range=(1, 2) means use both single words AND 2-word pairs
vec_fake = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1, 2))
X_vec = vec_fake.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Logistic Regression is often superior for text classification
model_fake = LogisticRegression(max_iter=1000) 
model_fake.fit(X_train, y_train)

y_pred = model_fake.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Fake/Real Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

joblib.dump(model_fake, "model_fake.pkl")
joblib.dump(vec_fake, "vectorizer_fake.pkl")


# --- MODEL 2: TOPIC CLASSIFIER (Upgraded) ---
print("\n--- Training Topic Model (Logistic Regression) ---")
df_topic = pd.read_csv("topic_training_data.csv").dropna()
X_topic = df_topic['text']
y_topic = df_topic['topic']

vec_topic = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1, 2))
X_vec_t = vec_topic.fit_transform(X_topic)

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_vec_t, y_topic, test_size=0.2, random_state=42)

model_topic = LogisticRegression(max_iter=1000)
model_topic.fit(X_train_t, y_train_t)

y_pred_t = model_topic.predict(X_test_t)
acc_t = accuracy_score(y_test_t, y_pred_t)
print(f"Topic Accuracy: {acc_t:.4f}")

joblib.dump(model_topic, "model_topic.pkl")
joblib.dump(vec_topic, "vectorizer_topic.pkl")

print("\n✅ Upgraded models saved successfully.")