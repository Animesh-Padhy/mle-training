import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import logging


LOG_DIR = os.path.join("../..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "train_data.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


def train_model(input_folder, output_folder):
    try:
        housing_path = os.path.join(input_folder, "housing.csv")
        housing = pd.read_csv(housing_path)

        train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

        housing["income_cat"] = pd.cut(
            housing["median_income"],
            bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
            labels=[1, 2, 3, 4, 5],
        )

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]

        train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

        compare_props = pd.DataFrame(
            {
                "Overall": income_cat_proportions(housing),
                "Stratified": income_cat_proportions(strat_test_set),
                "Random": income_cat_proportions(test_set),
            }
        ).sort_index()
        compare_props["Rand. %error"] = (
            100 * compare_props["Random"] / compare_props["Overall"] - 100
        )
        compare_props["Strat. %error"] = (
            100 * compare_props["Stratified"] / compare_props["Overall"] - 100
        )

        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)

        housing = strat_train_set.copy()
        housing.plot(kind="scatter", x="longitude", y="latitude")
        housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

        housing1 = housing.drop("ocean_proximity", axis=1)
        corr_matrix = housing1.corr()
        corr_matrix["median_house_value"].sort_values(ascending=False)
        housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
        housing["bedrooms_per_room"] = (
            housing["total_bedrooms"] / housing["total_rooms"]
        )
        housing["population_per_household"] = (
            housing["population"] / housing["households"]
        )

        housing = strat_train_set.drop("median_house_value", axis=1)
        housing_labels = strat_train_set["median_house_value"].copy()

        imputer = SimpleImputer(strategy="median")

        housing_num = housing.drop("ocean_proximity", axis=1)

        imputer.fit(housing_num)
        X = imputer.transform(housing_num)

        housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
        housing.loc["rooms_per_household"] = (
            housing_tr["total_rooms"] / housing_tr["households"]
        )
        housing.loc["bedrooms_per_room"] = (
            housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
        )
        housing.loc["population_per_household"] = (
            housing_tr["population"] / housing_tr["households"]
        )

        housing_cat = housing[["ocean_proximity"]]
        housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

        lin_reg = LinearRegression()
        lin_reg.fit(housing_prepared, housing_labels)

        housing_predictions = lin_reg.predict(housing_prepared)
        lin_mse = mean_squared_error(housing_labels, housing_predictions)
        lin_rmse = np.sqrt(lin_mse)
        lin_rmse

        lin_mae = mean_absolute_error(housing_labels, housing_predictions)
        lin_mae

        tree_reg = DecisionTreeRegressor(random_state=42)
        tree_reg.fit(housing_prepared, housing_labels)

        housing_predictions = tree_reg.predict(housing_prepared)
        tree_mse = mean_squared_error(housing_labels, housing_predictions)
        tree_rmse = np.sqrt(tree_mse)
        tree_rmse

        param_distribs = {
            "n_estimators": randint(low=1, high=200),
            "max_features": randint(low=1, high=8),
        }

        forest_reg = RandomForestRegressor(random_state=42)
        rnd_search = RandomizedSearchCV(
            forest_reg,
            param_distributions=param_distribs,
            n_iter=10,
            cv=5,
            scoring="neg_mean_squared_error",
            random_state=42,
        )
        rnd_search.fit(housing_prepared, housing_labels)
        cvres = rnd_search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)

        param_grid = [
            {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
            {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
        ]

        forest_reg = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            forest_reg,
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            return_train_score=True,
        )
        grid_search.fit(housing_prepared, housing_labels)

        # handler.setFormatter(formatter)
        # logger.addHandler(handler)
        # logger.setLevel(logging.DEBUG)

        grid_search.best_params_
        cvres = grid_search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)

        feature_importances = grid_search.best_estimator_.feature_importances_
        sorted(zip(feature_importances, housing_prepared.columns), reverse=True)

        final_model = grid_search.best_estimator_

        os.makedirs(output_folder, exist_ok=True)
        model_path = os.path.join(output_folder, "trained_model.pkl")
        joblib.dump(final_model, model_path)
        logging.info("Model Trained.")
    except Exception as e:
        logging.error(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("input_folder", help="Path to the input dataset folder.")
    parser.add_argument(
        "output_folder",
        nargs="?",
        default="../model/",
        help="Path to the output model folder.",
    )
    args = parser.parse_args()
    logging.info("Started training model.")
    train_model(args.input_folder, args.output_folder)
    logging.info("Finished training model.")


if __name__ == "__main__":
    main()
