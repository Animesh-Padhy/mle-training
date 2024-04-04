import subprocess


def main():
    subprocess.run(
        [
            "python",
            "src/house_price_prediction/ingest_data.py",
            "data/housing",
        ]
    )
    subprocess.run(
        [
            "python",
            "src/house_price_prediction/train.py",
            "data/housing",
        ]
    )
    subprocess.run(
        [
            "python",
            "src/house_price_prediction/score.py",
            "artifacts/models/final_model.pkl",
            "data/housing",
        ]
    )


if __name__ == "__main__":
    main()
