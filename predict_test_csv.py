import pandas as pd
import joblib
import json
from datetime import datetime

model = joblib.load("./model/fake_news_model.joblib")

df = pd.read_csv("./data/fake_or_real_news.csv")

json_data = df.to_dict(orient="records")

# opcional: guardar el JSON físico
with open("dataset.json", "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=4, ensure_ascii=False)

results = []

for item in json_data:

    text_input = str(item["text"])

    pred = model.predict([text_input])[0]

    if hasattr(model, "predict_proba"):
        confidence = float(max(model.predict_proba([text_input])[0]))
    else:
        confidence = 0.5

    result = {
        "title": item["title"],
        "source": item.get("source", "unknown"),
        "is_fake": bool(pred),
        "confidence": round(confidence, 2),
        "timestamp": datetime.now().isoformat()
    }

    results.append(result)

with open("predictions.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(json.dumps(results[0], indent=4, ensure_ascii=False))