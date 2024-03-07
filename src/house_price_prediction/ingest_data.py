import argparse
import logging
import os
import tarfile

import pandas as pd
import six

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
LOGS_PATH = os.path.join(LOGS_DIR, "program_logs.log")
logging.basicConfig(
    level=logging.INFO,
    filename=LOGS_PATH,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_path, housing_url=HOUSING_URL):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    six.moves.urllib.request.urlretrieve(housing_url, tgz_path)
    logging.info("Fetched housing data successfully.")
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def main():
    parser = argparse.ArgumentParser(
        description="Takes Housing Path as input and returns a CSV file of the housing data."
    )
    parser.add_argument("housing_path", type=str, help="Enter Housing Path")
    args = parser.parse_args()
    housing_path = args.housing_path

    try:
        fetch_housing_data(housing_path=housing_path)
    except Exception as e:
        logging.error("Error", e)


if __name__ == "__main__":
    logging.info("Initiated execution of ingest_data.py file.")
    main()
