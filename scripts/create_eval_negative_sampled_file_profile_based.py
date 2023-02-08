import argparse
import csv
import math
import os
import random
from collections import Counter, defaultdict
import pandas as pd

ITEM_ID_FIELD = "item_id"
USER_ID_FIELD = "user_id"


def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier + 0.5) / multiplier


def load_data(dataset_path):
    ret = defaultdict()
    for sp in ["train", "validation", "test"]:
        ret[sp] = pd.read_csv(os.path.join(dataset_path, f"{sp}.csv"), dtype=str)
    return ret


def get_items_by_genre(dataset_path, genre_field):
    item_file = os.path.join(dataset_path, "items.csv")
    items_by_genre = defaultdict(list)
    item_genres = defaultdict(list)
    items = pd.read_csv(item_file, dtype=str)
    items[genre_field] = items[genre_field].fillna("")
    # some book do not  have any genre, these are considered as same genre! as we don't want to loose them in neg sampling
    for item_id, genres in zip(items[ITEM_ID_FIELD], items[genre_field]):
        if genre_field in ["category", "genres"]:
            for g in [g.replace("'", "").replace('"', "").replace("[", "").replace("]", "").strip() for g in genres.split(",")]:
                items_by_genre[g].append(item_id)
                item_genres[item_id].append(g)
        else:
            raise NotImplementedError()
    return items_by_genre, item_genres


def get_user_used_items(datasets):
    used_items = defaultdict(lambda: defaultdict(set))
    for split in datasets.keys():
        for user_iid, item_iid in zip(datasets[split][USER_ID_FIELD], datasets[split][ITEM_ID_FIELD]):
            used_items[split][user_iid].add(item_iid)

    return used_items


def get_user_genre_cnt(data, item_genres):
    user_genres = defaultdict(lambda: defaultdict(lambda: 0))
    for user_id, item_id in zip(data[USER_ID_FIELD], data[ITEM_ID_FIELD]):
        for g in item_genres[item_id]:
            user_genres[user_id][g] += 1
    return user_genres


def neg_sampling_by_train_genre(pos_eval_data, per_user_train_genre_cnt, items_by_genre, used_items, num_neg_samples):
    neg_samples = []
    eval_user_counter = Counter(pos_eval_data[USER_ID_FIELD])
    for user_id in eval_user_counter.keys():
        num_pos = eval_user_counter[user_id]
        max_num_user_neg_samples = num_pos * num_neg_samples  # TODO min allitems?
        # get the propotions based on the genre count
        remaining_num_samples = max_num_user_neg_samples
        num_total_genres = sum(per_user_train_genre_cnt[user_id].values())
        user_samples = set()
        # first try getting number of samples per genre with max 1 to avoid having 0 at the begining
        num_samples_per_genre = defaultdict(lambda: 0)
        for g in per_user_train_genre_cnt[user_id]:
            num_samples_per_genre[g] = max(1, int(round_half_up((per_user_train_genre_cnt[user_id][
                                                         g] * remaining_num_samples) / num_total_genres)))
        outer_try = 0
        while remaining_num_samples > 0:
            if outer_try == 100:
                print(
                    f"WARM! cannot sample all needed samples-user:'{user_id}'. " \
                    f"Remaining needed samples {remaining_num_samples}/{max_num_user_neg_samples}")
                break
            outer_try += 1
            remaining_num_samples = remaining_num_samples - sum(num_samples_per_genre.values())
            # if we tried sampling more than needed reduce them from more frequent ones:
            while remaining_num_samples < 0:
                for g, sample_cnt in sorted(num_samples_per_genre.items(), key=lambda x: x[1], reverse=True):
                    if num_samples_per_genre[g] != 0:
                        num_samples_per_genre[g] -= 1
                        remaining_num_samples += 1
                    if remaining_num_samples == 0:
                        break

            for g, sample_cnt in sorted(num_samples_per_genre.items(), key=lambda x: x[1]):
                try_cnt = 0
                while sample_cnt > 0:
                    if try_cnt == 100:
                        # print(f"Warn! cannot sample all needed samples for genre '{g}' with total {len(items_by_genre[g])} items. Remaining needed samples {sample_cnt}/{num_samples_per_genre[g]}")
                        remaining_num_samples += sample_cnt
                        break
                    try_cnt += 1
                    current_samples = set(random.sample(items_by_genre[g], min(len(items_by_genre[g]), sample_cnt)))
                    current_samples -= user_samples
                    current_samples -= used_items[user_id]
                    sample_cnt -= len(current_samples)
                    user_samples.update(current_samples)

            # when we still have remaining: let's distribute them evenly from most freq to least instead, as otherwise it will be 0 for most cases
            num_samples_per_genre = defaultdict(lambda: 0)
            if remaining_num_samples > 0:
                for g in [g for g, v in sorted(per_user_train_genre_cnt[user_id].items(), key=lambda x: x[1], reverse=True)]:
                    num_samples_per_genre[g] += 1
                    if sum(num_samples_per_genre.values()) == remaining_num_samples:
                        break

        neg_samples.extend([[user_id, item_id, 0] for item_id in user_samples])
    return neg_samples


def main(dataset_path, eval_set, num_neg_samples, genre_field):
    datasets = load_data(dataset_path)
    item_by_genre, item_genres = get_items_by_genre(dataset_path, genre_field)
    per_user_train_genre_cnt = get_user_genre_cnt(datasets["train"], item_genres)
    user_used_items = get_user_used_items(datasets)

    # for validation: train and current positive items are used items.
    used_items = user_used_items['train'].copy()
    for user_id, cur_user_items in user_used_items['validation'].items():
        used_items[user_id] = used_items[user_id].union(cur_user_items)
    if eval_set == "test":
        for user_id, cur_user_items in user_used_items['test'].items():
            used_items[user_id] = used_items[user_id].union(cur_user_items)

    neg_samples = neg_sampling_by_train_genre(datasets[eval_set], per_user_train_genre_cnt, item_by_genre,
                                              used_items, num_neg_samples)
    with open(os.path.join(dataset_path, f'{eval_set}_negatives_profile_based_evaluation_{num_neg_samples}.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'item_id', 'label'])
        writer.writerows(neg_samples)
    print(f"neg sampling profile based for {eval_set} done")


if __name__ == "__main__":
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, help='path to dataset')
    parser.add_argument('--set', type=str, help='test or validation')
    parser.add_argument('--ns', type=int, help='number of negative samples')
    parser.add_argument('--gf', type=str, help='genre field')
    args, _ = parser.parse_known_args()

    if args.set not in ["validation", "test"]:
        raise ValueError(f"{args.set} given value is wrong.")

    main(args.dataset_folder, args.set, args.ns, args.gf)