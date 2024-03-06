import argparse
import os
import tarfile
from six.moves import urllib
import logging

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("data", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "data/housing/housing.tgz"


LOG_DIR = os.path.join("../..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "ingest_data.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def fetch_housing_data(housing_path):
    try:
        housing_url = HOUSING_URL
        os.makedirs(housing_path, exist_ok=True)
        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()
        logging.info("Data fetched successfully.")
    except Exception as e:
        logging.error(f"Error fetching data: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and create training and validation data."
    )
    parser.add_argument(
        "output_folder",
        nargs="?",
        default="../../data",
        help="Path to the output folder for data.",
    )
    args = parser.parse_args()
    logging.info("Started fetching data.")
    fetch_housing_data(os.path.join(args.output_folder, "housing"))
    logging.info("Finished fetching data.")


if __name__ == "__main__":
    main()
