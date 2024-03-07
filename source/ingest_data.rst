Ingest Data
===========

This document outlines the steps to ingest data for machine learning using the `ingest_data.py` script.

**Objective**: The `ingest_data.py` script aims to organize data to facilitate the training of machine learning models.

**Usage**: Execute the script with the input data folder path and optionally specify the output folder path.

Functionality
-------------

The `ingest_data` function in the script reads data and saves the processed data to the specified output folder.

- **ingest_data**:
  - Arguments:
    - `output_path` (str, optional): Path to the output folder to save the read data.
  - Returns: None

Logging
-------

The script logs information about the data ingestion process to a log file located in the "logs" directory.

