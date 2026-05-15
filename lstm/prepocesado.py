import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.utils import pad_sequences

import nltk
import regex as re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))

df_fake = pd.read_csv("./data/Fake.csv")
df_true = pd.read_csv("./data/True.csv")

def clean_true(df):
    df["text"] = df["text"].str.replace(r"^.*?-", "", regex=True)
    df["text"] = df["text"].str.replace(r"\s*\[[^\[]*$", "", regex=True)
    df["text"] = df["text"].str.replace(r"(?i)\breuters\b", "", regex=True)
    df["text"] = df["text"].str.replace(r"\S*bit\.ly\S*", "", regex=True)
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    return df

def clean_combined(df):
    df["text"] = df["text"].str.replace(r"\[.*?\]", "", regex=True)
    df["text"] = df["text"].str.replace(r"\S*www\.\S*", "", regex=True)
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    return df

df_true = clean_true(df_true)

df_true["label"] = 1
df_fake["label"] = 0

df = pd.concat([df_true, df_fake], ignore_index=True)
df = clean_combined(df)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df["max_length"] = df["text"].apply(lambda x: len(str(x).split()))
max_length = df["max_length"].max()

text_cleaning = r"\b0\S*|\b[^A-Za-z0-9]+"

def preprocess_filter(text, stem=False):
    text = re.sub(text_cleaning, " ", str(text).lower()).strip()
    tokens = []
    stemmer = SnowballStemmer("english")

    for token in text.split():
        if token not in stop_words:
            if stem:
                token = stemmer.stem(token)
            tokens.append(token)

    return " ".join(tokens)

def word_embedding(text):
    text = preprocess_filter(text)
    return one_hot(text, 5000)

X = df["text"].apply(word_embedding).values
X = pad_sequences(X, maxlen=40, padding='pre')

y = df["label"].values

model = Sequential()
model.add(Embedding(5000, 40, input_length=40))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=64)

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

print("Confusion matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report")
print(classification_report(y_test, y_pred))