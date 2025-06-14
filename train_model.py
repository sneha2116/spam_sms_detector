# train_model.py

import pandas as pd
import zipfile
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Unzip and Load the dataset
with zipfile.ZipFile('spam.csv.zip', 'r') as zip_ref:
    zip_ref.extractall()

# Load the extracted CSV (likely named 'spam.csv')
df = pd.read_csv('spam.csv', encoding='latin-1')

# 2. Preprocessing
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# Encode labels: 'ham' -> 0, 'spam' -> 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 3. Split into train and test
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# 4. Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)

# 5. Train a simple model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 6. Save the model and vectorizer
with open('spam_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("Model and Vectorizer saved successfully!")
