#### Data Preparation:

1- Download the Amazon Review Data (reviews ane metadata) for Books from https://nijianmo.github.io/amazon/index.html. You should have obtained "Books.json.gz" and "meta_Books.json.gz".

2- Obtain the data splits with user and item ids from https://personalization.mpi-inf.mpg.de/SBR_DATA/ (only train_ids.csv is required for step3).

3- Run the script SBR/scripts/prepare_dataset_files.py after editing it with the path pointing to the downloaded data. It should create files: train.csv, items.csv, users.csv

4- Move train.csv, items.csv, users.csv from step3 and validation.csv, test.csv, files containing negative samples for evaluation to the same directory.

