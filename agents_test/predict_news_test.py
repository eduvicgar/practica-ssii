import joblib


MODEL_PATH = "model/fake_news_model.joblib"

LABELS = {
    0: "TRUE",
    1: "FAKE"
}


def main():
    model = joblib.load(MODEL_PATH)

    print("Modelo cargado correctamente.")
    print("Pega una noticia para analizarla.")
    print("Cuando termines de pegarla, pulsa ENTER dos veces.")
    print("Escribe 'salir' para cerrar.\n")

    while True:
        lines = []

        print("Introduce la noticia:")

        while True:
            line = input()

            if line.lower() == "salir":
                return

            if line == "":
                break

            lines.append(line)

        text = " ".join(lines).strip()

        if not text:
            print("No has introducido ningún texto.\n")
            continue

        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]

        print("\nResultado:")
        print(f"Predicción: {LABELS[prediction]}")
        print(f"Probabilidad TRUE: {probabilities[0]:.2%}")
        print(f"Probabilidad FAKE: {probabilities[1]:.2%}")
        print("-" * 50)
        print()
        

if __name__ == "__main__":
    main()