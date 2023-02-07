#### Data Preparation:

1- Download the Amazon Review Data (reviews ane metadata) for Books from https://nijianmo.github.io/amazon/index.html. You should have obtained "Books.json.gz" and "meta_Books.json.gz".

2- Obtain the data splits with user and item ids from https://personalization.mpi-inf.mpg.de/SBR_DATA/ .

3- Run the script scripts/prepare_dataset_files.py after editing it with the path pointing to the downloaded data. It should create files: train.csv, items.csv, users.csv

4- You should have the following files in the same directory: train.csv, validation.csv, test.csv, items.csv, users.csv, and the files with negative samples.


