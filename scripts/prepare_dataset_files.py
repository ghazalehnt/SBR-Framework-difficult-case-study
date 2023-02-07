import csv
import gzip
import json
from collections import defaultdict
from os.path import join


def fill_train_set(original_data_file, train_set_ids, output_path):
    interactions = defaultdict(lambda: defaultdict())
    with open(train_set_ids, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        item_idx = header.index("item_id")
        user_idx = header.index("user_id")
        for line in reader:
            interactions[line[user_idx]][line[item_idx]] = line

    with gzip.open(original_data_file) as f:
        for line in f:
            r = json.loads(line)
            user_id = r[ORIGINAL_USER_ID_FIELD]
            item_id = r[ORIGINAL_ITEM_ID_FIELD]
            if user_id in interactions:
                if item_id in interactions[user_id]:
                    for field in ADDITIONAL_INTERACTION_FIELDS:
                        if field in r:
                            interactions[user_id][item_id].append(r[field].replace("'", "").replace('"', "").replace("\n", " "))
                        else:
                            interactions[user_id][item_id].append("")


    train_item_ids = []
    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        header.extend(ADDITIONAL_INTERACTION_FIELDS)
        writer.writerow(header)
        for user, items in interactions.items():
            train_item_ids.extend(items.keys())
            for item, line in items.items():
                writer.writerow(line)
    return set(train_item_ids), set(interactions.keys())


def get_user_item_ids(input_file):
    ret_item_ids = []
    ret_user_ids = []
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        item_idx = header.index("item_id")
        user_idx = header.index("user_id")
        for line in reader:
            ret_item_ids.append(line[item_idx])
            ret_user_ids.append(line[user_idx])
    return set(ret_item_ids), set(ret_user_ids)


def write_item_meta_file(item_metadata_file, output_file, all_item_ids):
    metadata = defaultdict()
    with gzip.open(item_metadata_file) as f:
        for line in f:
            r = json.loads(line)
            item_id = r[ORIGINAL_ITEM_ID_FIELD]
            if item_id in all_item_ids:
                metadata[item_id] = [item_id]
                for field in ADDITIONAL_ITEM_FIELDS:
                    if type(r[field]) == list:
                        metadata[item_id].append(", ".join([i.strip() for i in r[field] if i.strip() != ""]).replace("[", "").replace("]", "").replace("'", "").replace('"', "").replace("\n", " "))
                    else:
                        metadata[item_id].append(r[field].replace("[", "").replace("]", "").replace("'", "").replace('"', "").replace("\n", " "))


    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        header = ["item_id"]
        header.extend(ADDITIONAL_ITEM_FIELDS)
        writer.writerow(header)
        for item_id in all_item_ids:
            writer.writerow(metadata[item_id])


def write_user_meta_file(output_file, all_user_ids):
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        header = ["user_id"]
        writer.writerow(header)
        for user_id in all_user_ids:
            writer.writerow([user_id])


if __name__ == "__main__":
    # path which has the files "Books.json.gz" and "meta_Books.json.gz":
    ORIGINAL_DATASET_PATH = "TODO"

    # path which has the files train_ids.csv, validation.csv, test.csv, and negative sampling files.
    SPLIT_IDS_PATH = "TODO"

    # output path (set to path of downloaded splits, change if needed):
    OUTPUT_PATH = SPLIT_IDS_PATH

    # change for different datasets:
    ORIGINAL_USER_ID_FIELD = "reviewerID"
    ORIGINAL_ITEM_ID_FIELD = "asin"
    ADDITIONAL_INTERACTION_FIELDS = ["reviewText", "summary"]
    ADDITIONAL_ITEM_FIELDS = ["title", "description", "category", "brand"]

    item_ids, user_ids = fill_train_set(join(ORIGINAL_DATASET_PATH, "Books.json.gz"),
                                        join(SPLIT_IDS_PATH, "train_ids.csv"),
                                        join(OUTPUT_PATH, "train.csv"))

    temp_item_ids, temp_user_ids = get_user_item_ids(join(SPLIT_IDS_PATH, "validation.csv"))
    item_ids = item_ids.union(temp_item_ids)
    user_ids = user_ids.union(temp_user_ids)
    temp_item_ids, temp_user_ids = get_user_item_ids(join(SPLIT_IDS_PATH, "test.csv"))
    item_ids = item_ids.union(temp_item_ids)
    user_ids = user_ids.union(temp_user_ids)

    write_item_meta_file(join(ORIGINAL_DATASET_PATH, "meta_Books.json.gz"),
                         join(SPLIT_IDS_PATH, "items.csv"),
                         item_ids)

    write_user_meta_file(join(SPLIT_IDS_PATH, "users.csv"), user_ids)

