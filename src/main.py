import subprocess


def main():
    subprocess.run(
        ["python", "house_price_prediction/ingest_data.py", "../../datasets"]
    )
    subprocess.run(
        [
            "python",
            "house_price_prediction/train.py",
            "../../datasets/housing",
            "./model",
        ]
    )
    subprocess.run(
        [
            "python",
            "house_price_prediction/score.py",
            "./model/trained_model.pkl",
            "../../datasets/housing",
        ]
    )
    print("end")


if __name__ == "__main__":
    main()
