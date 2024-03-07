import unittest
import importlib.util


class TestPackageInstallation(unittest.TestCase):
    def test_import_numpy(self):
        self.assertTrue(
            importlib.util.find_spec("numpy") is not None, "numpy is not installed"
        )

    def test_import_pandas(self):
        self.assertTrue(
            importlib.util.find_spec("pandas") is not None, "pandas is not installed"
        )

    def test_import_sklearn(self):
        self.assertTrue(
            importlib.util.find_spec("sklearn") is not None,
            "scikit-learn is not installed",
        )

    def test_import_scipy(self):
        self.assertTrue(
            importlib.util.find_spec("scipy") is not None, "scipy is not installed"
        )

    def test_import_six(self):
        self.assertTrue(
            importlib.util.find_spec("six") is not None, "six is not installed"
        )

    def test_import_argparse(self):
        self.assertTrue(
            importlib.util.find_spec("argparse") is not None,
            "argparse is not installed",
        )

    def test_import_hpp(self):
        self.assertTrue(
            importlib.util.find_spec("house_price_prediction") is not None,
            "hpp is not installed",
        )


if __name__ == "__main__":
    unittest.main()
