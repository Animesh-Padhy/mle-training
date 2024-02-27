# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## To excute the script
python < scriptname.py >


## instructions on how to run the code.
- download the nonstandard.py file into the linux home directory
- create a new environment named 'mle-dev'
  (base) animesh45@TIGER04432:~$ conda create --name mle-dev
- activate the environment mle-env
  (base) animesh45@TIGER04432:~$ conda activate mle-dev
- run the python file
  (mle-dev) animesh45@TIGER04432:~$ python3 nonstandardcode.py
- export the environments file as env.yml
  (mle-dev) animesh45@TIGER04432:~$ conda env export --name mle-dev > env.yml

  
