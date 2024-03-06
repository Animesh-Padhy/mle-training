import unittest
import os

from house_price_prediction import train


class TestTrain(unittest.TestCase):
    def test_train_model(self):

        input_folder = "../../data/housing"
        output_folder = "./model"
        os.makedirs(output_folder, exist_ok=True)
        train.train_model(input_folder, output_folder)
        file_path = os.path.join(output_folder, "trained_model.pkl")
        self.assertTrue(os.path.exists(file_path))


if __name__ == "__main__":
    unittest.main()
