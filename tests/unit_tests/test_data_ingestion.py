import unittest
import os
import pandas as pd

from house_price_prediction import ingest_data


class TestDataIngestion(unittest.TestCase):
    def test_fetch_housing_data(self):
        output_folder = "../../datasets"
        os.makedirs(output_folder, exist_ok=True)
        ingest_data.fetch_housing_data(output_folder)
        file_path = os.path.join(output_folder, "housing", "housing.csv")
        self.assertTrue(os.path.exists(file_path))
        data = pd.read_csv(file_path)
        self.assertIsInstance(data, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
