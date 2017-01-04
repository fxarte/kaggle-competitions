#!/bin/bash 

mysqlimport --ignore-lines=1  --fields-terminated-by=, --local -u root -p kaggle-bosch data/train_numeric.csv

