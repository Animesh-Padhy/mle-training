import unittest
from house_price_prediction import score
import pandas as pd
from sklearn.impute import SimpleImputer


class TestScore(unittest.TestCase):
    def test_evaluate_model(self):

        model_path = "./model/trained_model.pkl"
        input_folder = "data/housing/housing.csv"
        housing = pd.read_csv(input_folder)
        housing_labels = housing["median_house_value"].copy()
        housing = housing.drop("median_house_value", axis=1)

        imputer = SimpleImputer(strategy="median")
        housing_num = housing.drop("ocean_proximity", axis=1)
        imputer.fit(housing_num)
        X = imputer.transform(housing_num)
        housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
        housing_cat = housing[["ocean_proximity"]]
        housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))
        score.evaluate_model(model_path, housing_prepared, housing_labels)
        # add logic


if __name__ == "__main__":
    unittest.main()
