import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from tqdm import tqdm


CSV_PATH = "data/fake_or_real_news.csv"
MODEL_PATH = "model/fake_news_model.joblib"


def load_data():
    steps = tqdm(total=4, desc="Cargando datos", unit="paso")

    df = pd.read_csv(CSV_PATH)
    steps.update(1)

    if "title" in df.columns and "text" in df.columns:
        df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    elif "text" in df.columns:
        df["content"] = df["text"].fillna("")
    elif "content" in df.columns:
        df["content"] = df["content"].fillna("")
    else:
        raise ValueError("No hay columna válida de texto: usa 'text', 'title' o 'content'.")

    steps.update(1)

    if "label" not in df.columns:
        raise ValueError("No encuentro la columna 'label' en el CSV.")

    df["label"] = df["label"].astype(str).str.upper().str.strip()

    label_map = {
        "REAL": 0,
        "TRUE": 0,
        "0": 0,
        "FAKE": 1,
        "FALSE": 1,
        "1": 1
    }

    df["label"] = df["label"].map(label_map)

    if df["label"].isna().any():
        raise ValueError("Hay etiquetas no válidas en el CSV.")

    df["label"] = df["label"].astype(int)
    steps.update(1)

    X = df["content"]
    y = df["label"]
    steps.update(1)

    steps.close()

    return X, y


def show_most_important_words(model, top_n=20):
    feature_names = model.named_steps["tfidf"].get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[0]

    top_fake = coefficients.argsort()[-top_n:]
    top_real = coefficients.argsort()[:top_n]

    print("\n" + "=" * 60)
    print(f"Top {top_n} palabras o expresiones más asociadas a FAKE")
    print("=" * 60)

    for i in reversed(top_fake):
        print(f"{feature_names[i]:30s} {coefficients[i]:.4f}")

    print("\n" + "=" * 60)
    print(f"Top {top_n} palabras o expresiones más asociadas a REAL")
    print("=" * 60)

    for i in top_real:
        print(f"{feature_names[i]:30s} {coefficients[i]:.4f}")

    print()


def train():
    steps = tqdm(total=6, desc="Entrenamiento completo", unit="paso")

    X, y = load_data()
    steps.update(1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    steps.update(1)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_df=0.7,
            min_df=2,
            ngram_range=(1, 2)
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            verbose=1
        ))
    ])
    steps.update(1)

    model.fit(X_train, y_train)
    steps.update(1)

    y_pred = model.predict(X_test)
    steps.update(1)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))

    show_most_important_words(model, top_n=20)

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    steps.update(1)

    steps.close()

    print(f"Modelo guardado en {MODEL_PATH}")


if __name__ == "__main__":
    train()