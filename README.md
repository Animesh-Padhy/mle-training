# mle-training
## Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

### Scripts Overview
- ingest_data.py: Downloads and creates training and validation datasets. Accepts the output folder/file path as a user argument.
- train.py: Trains the model(s). Accepts arguments for input (dataset) and output folders (model pickles).
- score.py: Scores the model(s). Accepts arguments for model folder, dataset folder, and any outputs.

### Testing
Basic tests are provided to verify correct installation. Unit tests and functional tests are included for the library.

### Documentation
The code is documented using Sphinx to generate HTML documents. Docstrings follow the Numpy docstring style.

### Sklearn Pipeline Integration
The ML code has been refactored to use sklearn pipelines. Custom transformers have been created for new features generated in the code, such as rooms_per_household, bedrooms_per_room, and population_per_household.

### MLflow Integration
MLflow is utilized to track parameters and metrics in the data preparation, model training, and model scoring scripts. A main script orchestrates the tasks under a single parent MLflow run-id, with each child task having its own MLflow run-id.

 - utilized standard Python tools and best practices to create distribution archives for our Python package.
- Followed the Python packaging tutorial to generate distribution artifacts.
- Created Wheel (.whl) and Source distribution (tar.gz) files.
- Configured setuptools according to best practices for packaging Python projects.

### download the env.yml 
to run the script use the mle-dev enviroment

### To excute the script
python < src/main.py >
