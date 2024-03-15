import argparse
import os
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
import numpy as np
import logging
import mlflow

LOG_DIR = os.path.join("./", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "score_data.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_housing_data(housing_path):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def evaluate_model(model, housing_path, experiment_name):

    mlflow.set_experiment(experiment_name)

    temp1 = {}
    try:
        model = joblib.load(model)
        housing_path = os.path.join(housing_path, "housing.csv")
        housing = pd.read_csv(housing_path)

        housing_labels = housing["median_house_value"].copy()
        housing = housing.drop("median_house_value", axis=1)

        predictions = model.predict(housing)
        mse = mean_squared_error(housing_labels, predictions)
        rmse = np.sqrt(mse)
        print(f"Root Mean Squared Error: {rmse}")
        print(f"Mean Squared Error: {mse}")
        logging.info(f"Scoring successful. Root Mean Squared Error: {rmse}")

        # Log artifacts, metrics, and model
        mlflow.log_artifact(LOG_FILE)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mse", mse)

        temp1 = {"mse": mse, "rmse": rmse}
        return temp1

    except Exception as e:
        logging.error(f"Error: {e}")


def main():
    global housing_labels
    parser = argparse.ArgumentParser(description="Score the model.")
    parser.add_argument("model_path", help="Path to the trained model file.")
    parser.add_argument("input_folder", help="Path to the input dataset folder.")
    args = parser.parse_args()

    experiment_name = "HousePricePrediction"
    logging.info("Started scoring")
    evaluate_model(args.model_path, args.input_folder, experiment_name)
    logging.info("Finished scoring")


if __name__ == "__main__":
    main()
