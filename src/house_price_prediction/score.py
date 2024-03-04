import argparse
import os
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.impute import SimpleImputer
import logging

LOG_DIR = os.path.join("../..", "logs")
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


def evaluate_model(model, X, y):
    try:
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        print(f"Root Mean Squared Error: {rmse}")
        logging.info(f"Scoring successfully. Root Mean Squared Error: {rmse}")
    except Exception as e:
        logging.error(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Score the model.")
    parser.add_argument("model_path", help="Path to the trained model file.")
    parser.add_argument("input_folder", help="Path to the input dataset folder.")
    args = parser.parse_args()

    model = joblib.load(args.model_path)
    housing_path = os.path.join(args.input_folder, "housing.csv")
    housing = pd.read_csv(housing_path)
    housing_labels = housing["median_house_value"].copy()
    housing = housing.drop("median_house_value", axis=1)

    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))
    logging.info("Started scoring")
    evaluate_model(model, housing_prepared, housing_labels)
    logging.info("Finished scoring")


if __name__ == "__main__":
    main()
