import argparse
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


from sklearn.base import BaseEstimator, TransformerMixin


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        rooms_per_household = X[:, 3] / X[:, 6]
        bedrooms_per_room = X[:, 4] / X[:, 3]
        population_per_household = X[:, 5] / X[:, 6]
        return np.c_[
            X, rooms_per_household, bedrooms_per_room, population_per_household
        ]


LOG_DIR = os.path.join(".", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "train_data.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_housing_data(housing_path):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def train_model(input_folder, output_folder):
    housing = load_housing_data(input_folder)
    housing_labels = housing["median_house_value"].copy()
    housing = housing.drop("median_house_value", axis=1)

    # Define numeric and categorical columns
    numeric_features = housing.select_dtypes(include=[np.number]).columns
    categorical_features = housing.select_dtypes(include=[object]).columns

    # Define preprocessing pipelines for numeric and categorical data
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("attribs_adder", CombinedAttributesAdder()),  # Custom transformer
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder()),
        ]
    )

    # Combine preprocessing pipelines using ColumnTransformer
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    # Apply preprocessing pipeline to the input data
    housing_prepared = preprocessor.fit_transform(housing)

    # Define the parameter grid for grid search
    param_grid = [
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    # Train the model using grid search
    forest_reg = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing_prepared, housing_labels)

    # Print the grid search results
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    # Get the best model from grid search
    final_model = grid_search.best_estimator_

    # Save the trained model
    os.makedirs(output_folder, exist_ok=True)
    model_path = os.path.join(output_folder, "trained_model.pkl")
    joblib.dump(final_model, model_path)


def main():
    """
    Main function to train the model.
    """
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("input_folder", help="Path to the input dataset folder.")
    parser.add_argument(
        "output_folder",
        nargs="?",
        default="../model/",
        help="Path to the output model folder.",
    )
    args = parser.parse_args()

    train_model(args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()
