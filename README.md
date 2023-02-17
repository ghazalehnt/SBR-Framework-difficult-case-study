#### Data Preparation:

1- Download the Amazon Review Data (reviews ane metadata) for Books from https://nijianmo.github.io/amazon/index.html. You should have obtained "Books.json.gz" and "meta_Books.json.gz".

2- Obtain the data splits with user and item ids from https://personalization.mpi-inf.mpg.de/SBR_DATA/ .

3- Run the script scripts/prepare_dataset_files.py after editing it with the path pointing to the downloaded data. It should create files: train.csv, items.csv, users.csv

4- You should have the following files in the same directory: train.csv, validation.csv, test.csv, items.csv, users.csv, and the files with negative samples.


#### Pre-compute user and item textual representations:
After editing configs/precompute.json file by adding the dataset_path, run the following python scripts:
python precompute_user_item_chunk_reps_no_prec.py --config_file configs/precompute.json --which user;
python precompute_user_item_chunk_reps_no_prec.py --config_file configs/precompute.json --which item

For BERT+CF model, you need to have a trained CF model and edit config/precompute_cf.json and run the scripts using the mentioned config file.

#### Train:
After editing configs/bert5_uniform.json or other config files with different training negative sampling, run:
python main.py --config_file configs/configs/bert5_uniform.json --op train;

#### Evaluate:
