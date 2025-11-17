import pandas as pd
import numpy as np
import os

if not os.path.exists("Fake.csv"):
    print("Error: Fake.csv not found. Please upload it.")
else:
    df = pd.read_csv("Fake.csv")
    # Sample 2000 rows
    indices = np.linspace(0, len(df) - 1, 2000, dtype=int)
    subset = df.iloc[indices][["title"]].copy()
    
    subset = subset.rename(columns={"title": "text"})
    subset["label"] = "fake"
    # We don't add 'topic' here because we don't know it. 
    # The topic model will be trained only on Real news.

    subset.to_csv("fake_news.csv", index=False)
    print("Created fake_news.csv with 2000 rows")