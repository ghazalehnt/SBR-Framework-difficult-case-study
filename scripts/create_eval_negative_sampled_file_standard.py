import argparse
import csv
import os
import random
import time
from collections import Counter, defaultdict
import pandas as pd

ITEM_ID_FIELD = "item_id"
USER_ID_FIELD = "user_id"


def load_data(dataset_path):
    ret = defaultdict()
    for sp in ["train", "validation", "test"]:
        ret[sp] = pd.read_csv(os.path.join(dataset_path, f"{sp}.csv"), dtype=str)
    return ret


def get_user_used_items(datasets):
    used_items = defaultdict(lambda: defaultdict(set))
    for split in datasets.keys():
        for user_iid, item_iid in zip(datasets[split][USER_ID_FIELD], datasets[split][ITEM_ID_FIELD]):
            used_items[split][user_iid].add(item_iid)

    return used_items


def neg_sampling(data, used_items, strategy, num_neg_samples):
    all_items = []
    for items in used_items.values():
        all_items.extend(items)
    all_items = set(all_items)
    samples = []
    user_counter = Counter(data['user_id'])
    cnt = 1
    start_time = time.time()
    for user_id in user_counter.keys():
        if cnt % 100000 == 0:
            print(f"{cnt} users done")
        num_pos = user_counter[user_id]
        num_user_neg_samples = num_pos * num_neg_samples
        potential_items = list(all_items - used_items[user_id])
        if len(potential_items) < num_user_neg_samples:
            print(f"WARNING: as there were not enough potential items to sample for user {user_id} with "
                  f"{num_pos} positives needing {num_user_neg_samples} negs,"
                  f"we reduced the number of user negative samples to potential items {len(potential_items)}"
                  f"HOWEVER, bear in mind that this is problematic, as the validation has 0s for 1s of test!")
            num_user_neg_samples = len(potential_items)
        if strategy == 'random':
            for sampled_item in random.sample(potential_items, num_user_neg_samples):
                samples.append([user_id, sampled_item, 0])
        cnt += 1
    print(time.time()-start_time)
    return samples


def neg_sampling_opt(data, used_items, num_neg_samples):
    all_items = []  # ????
    for items in used_items.values():
        all_items.extend(items)
    all_items = list(set(all_items))
    samples = []
    user_counter = Counter(data['user_id'])
    user_cnt = 1
    start_time = time.time()
    for user_id in user_counter.keys():
        num_pos = user_counter[user_id]
        max_num_user_neg_samples = min(len(all_items), num_pos * num_neg_samples)
        if max_num_user_neg_samples < num_pos * num_neg_samples:
            print(f"WARN: user {user_id} needed {num_pos * num_neg_samples} samples,"
                  f"but all_items are {len(all_items)}")

        user_samples = set()
        try_cnt = -1
        num_user_neg_samples = max_num_user_neg_samples
        while True:
            if try_cnt == 100:
                print(f"WARN: After {try_cnt} tries, could not find {max_num_user_neg_samples} samples for"
                      f"{user_id}. We instead have {len(user_samples)} samples.")
                break
            current_samples = set(random.sample(all_items, num_user_neg_samples))
            current_samples -= user_samples
            cur_used_samples = used_items[user_id].intersection(current_samples)
            if len(user_samples) < max_num_user_neg_samples:
                current_samples = current_samples - cur_used_samples
                user_samples = user_samples.union(current_samples)
                num_user_neg_samples = max(max_num_user_neg_samples - len(user_samples), 0)
                # to make the process faster
                if num_user_neg_samples < len(user_samples):
                    num_user_neg_samples = min(max_num_user_neg_samples, num_user_neg_samples * 2)
                try_cnt += 1
            else:
                user_samples = user_samples.union(current_samples)
                if len(user_samples) > max_num_user_neg_samples:
                    user_samples = set(list(user_samples)[:max_num_user_neg_samples])
                break
        samples.extend([[user_id, sampled_item_id, 0] for sampled_item_id in user_samples])
        user_cnt += 1
    print(f"{user_cnt} users in {time.time()-start_time}")
    return samples


def main(dataset_path, eval_set, num_neg_samples):
    datasets = load_data(dataset_path)
    user_used_items = get_user_used_items(datasets)

    used_items = user_used_items['train'].copy()
    for user_id, cur_user_items in user_used_items['validation'].items():
        used_items[user_id] = used_items[user_id].union(cur_user_items)

    if eval_set == "test":
        for user_id, cur_user_items in user_used_items['test'].items():
            if user_id in used_items:
                used_items[user_id] = used_items[user_id].union(cur_user_items)
            else:
                used_items[user_id] = set()

    neg_samples = neg_sampling_opt(datasets[eval_set], used_items, num_neg_samples)
    with open(os.path.join(dataset_path, f'{eval_set}_negatives_standard_evaluation_{num_neg_samples}.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'item_id', 'label'])
        writer.writerows(neg_samples)
    print(f"neg sampling standard for {eval_set} done")


if __name__ == "__main__":
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_folder', type=str, help='path to dataset')
    parser.add_argument('set', type=str, help='test or validation')
    parser.add_argument('ns', type=int, help='number of negative samples')
    args, _ = parser.parse_known_args()

    if args.set not in ["validation", "test"]:
        raise ValueError(f"{args.set} given value is wrong.")

    main(args.dataset_folder, args.set, args.ns)
