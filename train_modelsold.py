import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- MODEL 1: FAKE vs REAL ---
print("\n--- Training Fake/Real Model ---")
df = pd.read_csv("fakereal_training_data.csv").dropna()
X = df['text']
y = df['label']

vec_fake = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vec_fake.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

model_fake = MultinomialNB()
model_fake.fit(X_train, y_train)
print(f"Fake/Real Accuracy: {accuracy_score(y_test, model_fake.predict(X_test)):.4f}")

joblib.dump(model_fake, "model_fake.pkl")
joblib.dump(vec_fake, "vectorizer_fake.pkl")


# --- MODEL 2: TOPIC CLASSIFIER ---
print("\n--- Training Topic Model ---")
df_topic = pd.read_csv("topic_training_data.csv").dropna()
X_topic = df_topic['text']
y_topic = df_topic['topic']

vec_topic = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec_t = vec_topic.fit_transform(X_topic)
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_vec_t, y_topic, test_size=0.2, random_state=42)

model_topic = MultinomialNB()
model_topic.fit(X_train_t, y_train_t)
print(f"Topic Accuracy: {accuracy_score(y_test_t, model_topic.predict(X_test_t)):.4f}")

joblib.dump(model_topic, "model_topic.pkl")
joblib.dump(vec_topic, "vectorizer_topic.pkl")

print("\n✅ All models saved successfully.")