## group users and items based on their count in training data and evaluate all ALL, and COLD, HOT separately.
# what is saved?
# dict user->item->score
# create different things to use metric functions
import argparse
import csv
import json
import os
import time
from collections import Counter, defaultdict

import transformers
import pandas as pd
import numpy as np

from SBR.utils.metrics import calculate_ranking_metrics_detailed

relevance_level = 1
prediction_threshold = 0.5

goodreads_rating_mapping = {
    None: None,  ## this means there was no rating
    'did not like it': 1,
    'it was ok': 2,
    'liked it': 3,
    'really liked it': 4,
    'it was amazing': 5
}


def group_users(config, thresholds, min_user_review_len=None, review_field=None):
    # here we have some users who only exist in training set
    split_datasets = defaultdict()
    for sp in ["train", "validation", "test"]:
        split_datasets[sp] = pd.read_csv(os.path.join(config['dataset']['dataset_path'], f"{sp}.csv"), dtype=str)

        if config['dataset']["name"] == "CGR":
            for k, v in goodreads_rating_mapping.items():
                split_datasets[sp]['rating'] = split_datasets[sp]['rating'].replace(k, v)
        elif config['dataset']["name"] == "GR_UCSD":
            split_datasets[sp]['rating'] = split_datasets[sp]['rating'].astype(int)
        elif config['dataset']["name"] == "Amazon":
            split_datasets[sp]['rating'] = split_datasets[sp]['rating'].astype(float).astype(int)
        else:
            raise NotImplementedError(f"dataset {config['dataset']['name']} not implemented!")

    if not config['dataset']['binary_interactions']:
        # if predicting rating: remove the not-rated entries and map rating text to int
        split_datasets = split_datasets.filter(lambda x: x['rating'] is not None)

    train_user_count = Counter(split_datasets['train']['user_id'])

    # here we have users with long reviews and rest would neet to be intersect with it
    if min_user_review_len is not None:
        # don't calc for limited train data as they are not comparable at this point
        if 'limit_training_data' in config['dataset'] and config['dataset']['limit_training_data'] != "":
            return {}, {}, set()
        
        split_datasets['train'][review_field] = split_datasets['train'][review_field].fillna("")
        keep_users = {}
        tokenizer = transformers.AutoTokenizer.from_pretrained(BERTMODEL)
        user_reviews = {}
        for user_id, review in zip(split_datasets['train']['user_id'], split_datasets['train'][review_field]):
            if user_id not in user_reviews:
                user_reviews[user_id] = []
            if review is not None:
                user_reviews[user_id].append(review)
        for user_id in user_reviews:
            user_reviews[user_id] = ". ".join(user_reviews[user_id])
            num_toks = len(tokenizer(user_reviews[user_id], truncation=False)['input_ids'])
            if num_toks >= min_user_review_len:
                keep_users[user_id] = num_toks
        # print(keep_users)
        train_user_count = {k: v for k, v in train_user_count.items() if k in keep_users.keys()}

    eval_users = set(split_datasets['test']['user_id'])
    eval_users.update(set(split_datasets['validation']['user_id']))
    eval_users = eval_users.intersection(train_user_count.keys())
    train_user_count_longtail = {str(k): v for k, v in train_user_count.items() if k not in eval_users}

    groups = {thr: set() for thr in sorted(thresholds)}
    if len(thresholds) > 0:
        groups['rest'] = set()
        for user in eval_users:
            added = False
            for thr in sorted(thresholds):
                if train_user_count[user] <= thr:
                    groups[thr].add(str(user))
                    added = True
                    break
            if not added:
                groups['rest'].add(str(user))

    ret_group = {}
    last = 1
    for gr in groups:
        if gr == 'rest':
            new_gr = f"{last}+"
        else:
            new_gr = f"{last}-{gr}"
            last = gr + 1
        ret_group[new_gr] = groups[gr]

    train_user_count = {str(k): v for k, v in train_user_count.items()}
    return ret_group, train_user_count, train_user_count_longtail


def get_metrics(ground_truth, prediction_scores, weighted_labels, ranking_metrics=None):
    if len(ground_truth) == 0:
        return {}, {}
    start = time.time()
    results = calculate_ranking_metrics_detailed(gt=ground_truth, pd=prediction_scores,
                                                             relevance_level=relevance_level,
                                                             given_ranking_metrics=ranking_metrics,
                                                             calc_pytrec=not weighted_labels)
    # micro avg:
    micro_res = defaultdict()
    for m in results:
        assert len(results[m]) == len(ground_truth)
        micro_res[m] = np.array(results[m]).mean().tolist()
    # macro avg per user:
    macro_res = defaultdict()
    for m in results:
        user_qids_res = defaultdict(list)
        for qid, r in zip(ground_truth.keys(), results[m]):
            user_id = qid[:qid.index("_")]
            user_qids_res[user_id].append(r)
        user_res = []
        for user_id in user_qids_res:
            user_res.append(np.array(user_qids_res[user_id]).mean().tolist())
        macro_res[m] = np.array(user_res).mean().tolist()
    print(f"ranking metrics in {time.time() - start}")
    return micro_res, macro_res


def main(config, valid_gt, valid_pd, test_gt, test_pd, test_qid_items, valid_qid_items, thresholds,
         min_user_review_len=None, review_field=None, test_neg_st=None, valid_neg_st=None, ranking_metrics=None):
    # start = time.time()
    # user_groups, train_user_count, train_user_count_longtail = group_users(config, thresholds,
    #                                                                        min_user_review_len, review_field)
    # if len(train_user_count) == 0:
    #     return
    # print(f"grouped users in {time.time()-start}")

    outfname = f"results_th_{'_'.join([str(t) for t in thrs])}_v-{valid_neg_st}_t-{test_neg_st}"
    valid_fname = f"results_valid_th_{'_'.join([str(t) for t in thrs])}_{valid_neg_st}"
    test_fname = f"results_test_th_{'_'.join([str(t) for t in thrs])}_{test_neg_st}"
    if min_user_review_len is not None:
        outfname += f"_min_review_len_{min_user_review_len}"
        valid_fname += f"_min_review_len_{min_user_review_len}"
        test_fname += f"_min_review_len_{min_user_review_len}"
    if best_epoch is not None:
        outfname += f"_epoch-{best_epoch}"
        valid_fname += f"_epoch-{best_epoch}"
        test_fname += f"_epoch-{best_epoch}"
    outfname += ".txt"
    valid_fname += ".csv"
    test_fname += ".csv"
    print(outfname)
    outf = open(os.path.join(result_folder, outfname), 'w')
    valid_csv_f = open(os.path.join(result_folder, valid_fname), "w")
    test_csv_f = open(os.path.join(result_folder, test_fname), "w")

    # why?
    # start = time.time()
    # test_gt = {k: v for k, v in test_gt.items() if k in train_user_count.keys()}
    # test_pd = {k: v for k, v in test_pd.items() if k in train_user_count.keys()}
    # valid_gt = {k: v for k, v in valid_gt.items() if k in train_user_count.keys()}
    # valid_pd = {k: v for k, v in valid_pd.items() if k in train_user_count.keys()}
    # print(f"? {time.time() - start}")
    #
    # if len(thresholds) > 0:
    #     assert sum([len(ug) for ug in user_groups.values()]) == len(set(test_gt.keys()).union(valid_gt.keys()))

    # let's count how many interactions are there
    start = time.time()
    # valid_pos_inter_cnt = {group: 0 for group in user_groups}
    # valid_neg_inter_cnt = {group: 0 for group in user_groups}
    valid_pos_total_cnt = 0
    valid_neg_total_cnt = 0
    for u in valid_gt:
        valid_pos_total_cnt += len([k for k, v in test_gt[u].items() if v == 1])
        valid_neg_total_cnt += len([k for k, v in test_gt[u].items() if v == 0])
        # group = [k for k in user_groups if u in user_groups[k]]
        # if len(group) == 0:
        #     continue
        # group = group[0]
        # valid_pos_inter_cnt[group] += len([k for k, v in valid_gt[u].items() if v == 1])
        # valid_neg_inter_cnt[group] += len([k for k, v in valid_gt[u].items() if v == 0])

    # test_pos_inter_cnt = {group: 0 for group in user_groups}
    # test_neg_inter_cnt = {group: 0 for group in user_groups}
    test_pos_total_cnt = 0
    test_neg_total_cnt = 0
    for u in test_gt:
        test_pos_total_cnt += len([k for k, v in test_gt[u].items() if v == 1])
        test_neg_total_cnt += len([k for k, v in test_gt[u].items() if v == 0])
        # group = [k for k in user_groups if u in user_groups[k]]
        # if len(group) == 0:
        #     continue
        # group = group[0]
        # test_pos_inter_cnt[group] += len([k for k, v in test_gt[u].items() if v == 1])
        # test_neg_inter_cnt[group] += len([k for k, v in test_gt[u].items() if v == 0])
    print(f"count inters {time.time() - start}")

    outf.write(f"#total_evaluation_users = {len(set(test_gt.keys()).union(valid_gt.keys()))} \n")
               # f"#total_training_users = {len(set(test_gt.keys()).union(valid_gt.keys())) + len(train_user_count_longtail)} \n"
               # f"#total_longtail_trainonly_users = {len(train_user_count_longtail)} \n")
    # for gr in user_groups:
    #     outf.write(f"#eval_user_group_{gr}: {len(user_groups[gr].intersection(set(test_gt.keys()).union(valid_gt.keys())))}  ")
    #     outf.write(f"#valid_user_group_{gr}: {len(user_groups[gr].intersection(valid_gt.keys()))}  ")
    #     outf.write(f"#test_user_group_{gr}: {len(user_groups[gr].intersection(test_gt.keys()))}\n")

    outf.write(f"#total_positive_inters_validation = {valid_pos_total_cnt}\n")
    # for gr in user_groups:
    #     outf.write(f"positive_inters_validation_user_group_{gr} = {valid_pos_inter_cnt[gr]}\n")
    outf.write(f"#total_negatove_inters_validation = {valid_neg_total_cnt}\n")
    # for gr in user_groups:
    #     outf.write(f"negative_inters_validation_user_group_{gr} = {valid_neg_inter_cnt[gr]}\n")

    outf.write(f"#total_positive_inters_test = {test_pos_total_cnt}\n")
    # for gr in user_groups:
    #     outf.write(f"positive_inters_test_user_group_{gr} = {test_pos_inter_cnt[gr]}\n")
    outf.write(f"#total_negatove_inters_test = {test_neg_total_cnt}\n")
    # for gr in user_groups:
    #     outf.write(f"negative_inters_test_user_group_{gr} = {test_neg_inter_cnt[gr]}\n")
    outf.write("\n\n")

    start = time.time()
    test_gt_per_qid = defaultdict(lambda: defaultdict())
    test_pd_per_qid = defaultdict(lambda: defaultdict())
    for user_id in test_qid_items:
        for ref_item in test_qid_items[user_id]:
            for item_id in test_qid_items[user_id][ref_item]:
                test_gt_per_qid[f"{user_id}_{ref_item}"][item_id] = test_gt[user_id][item_id]
                test_pd_per_qid[f"{user_id}_{ref_item}"][item_id] = test_pd[user_id][item_id]

    valid_gt_per_qid = defaultdict(lambda: defaultdict())
    valid_pd_per_qid = defaultdict(lambda: defaultdict())
    for user_id in valid_qid_items:
        for ref_item in valid_qid_items[user_id]:
            for item_id in valid_qid_items[user_id][ref_item]:
                valid_gt_per_qid[f"{user_id}_{ref_item}"][item_id] = valid_gt[user_id][item_id]
                valid_pd_per_qid[f"{user_id}_{ref_item}"][item_id] = valid_pd[user_id][item_id]
    print(f"create qids {time.time() - start}")

    rows_valid = []
    rows_test = []
    valid_results_micro, valid_results_macro = get_metrics(ground_truth=valid_gt_per_qid,
                                                           prediction_scores=valid_pd_per_qid,
                                                           weighted_labels=True if (valid_neg_st is not None and "-" in valid_neg_st) else False,
                                                           ranking_metrics=ranking_metrics)
    metric_header = sorted(valid_results_micro.keys())
    rows_valid.append(["group"] + metric_header)
    outf.write(f"MICRO Valid results ALL: {valid_results_micro}\n")
    rows_valid.append(["MICRO Valid - ALL"] + [valid_results_micro[h] for h in metric_header])
    metric_header = sorted(valid_results_macro.keys())
    rows_valid.append(["group"] + metric_header)
    outf.write(f"MACRO Valid results ALL: {valid_results_macro}\n")
    rows_valid.append(["MACRO Valid - ALL"] + [valid_results_macro[h] for h in metric_header])

    test_results_micro, test_results_macro = get_metrics(ground_truth=test_gt_per_qid,
                               prediction_scores=test_pd_per_qid,
                               weighted_labels=True if (test_neg_st is not None and "-" in test_neg_st) else False,
                               ranking_metrics=ranking_metrics)
    metric_header = sorted(test_results_micro.keys())
    rows_test.append(["group"] + metric_header)
    outf.write(f"MICRO Test results ALL: {test_results_micro}\n\n")
    rows_test.append(["MICRO Test - ALL"] + [test_results_micro[h] for h in metric_header])
    metric_header = sorted(test_results_macro.keys())
    rows_test.append(["group"] + metric_header)
    outf.write(f"MACRO Test results ALL: {test_results_macro}\n\n")
    rows_test.append(["MACRO Test - ALL"] + [test_results_macro[h] for h in metric_header])

    # GROUPS
    # for gr in user_groups:
    #     valid_results_micro, valid_results_macro = get_metrics(ground_truth={k: v for k, v in valid_gt_per_qid.items() if k[:k.index("_")] in user_groups[gr]},
    #                                 prediction_scores={k: v for k, v in valid_pd_per_qid.items() if k[:k.index("_")] in user_groups[gr]},
    #                                 weighted_labels=True if (valid_neg_st is not None and "-" in valid_neg_st) else False,
    #                                 ranking_metrics=ranking_metrics)
    #     metric_header = sorted(valid_results_micro.keys())
    #     outf.write(f"MICRO Valid results group: {gr}: {valid_results_micro}\n")
    #     rows_valid.append([f"MICRO Valid - group {gr}"] + [valid_results_micro[h] if h in valid_results_micro else "" for h in metric_header])
    #     metric_header = sorted(valid_results_macro.keys())
    #     outf.write(f"MACRO Valid results group: {gr}: {valid_results_macro}\n")
    #     rows_valid.append(
    #         [f"MACRO Valid - group {gr}"] + [valid_results_macro[h] if h in valid_results_macro else "" for h in
    #                                          metric_header])
    #
    #     test_results_micro, test_results_macro = get_metrics(ground_truth={k: v for k, v in test_gt_per_qid.items() if k[:k.index("_")] in user_groups[gr]},
    #                                prediction_scores={k: v for k, v in test_pd_per_qid.items() if k[:k.index("_")] in user_groups[gr]},
    #                                weighted_labels=True if (test_neg_st is not None and "-" in test_neg_st) else False,
    #                                ranking_metrics=ranking_metrics)
    #     metric_header = sorted(test_results_micro.keys())
    #     outf.write(f"MICRO Test results group: {gr}: {test_results_micro}\n\n")
    #     rows_test.append([f"MICRO Test - group {gr}"] + [test_results_micro[h] if h in test_results_micro else "" for h in metric_header])
    #     metric_header = sorted(test_results_macro.keys())
    #     outf.write(f"MACRO Test results group: {gr}: {test_results_macro}\n\n")
    #     rows_test.append(
    #         [f"MACRO Test - group {gr}"] + [test_results_macro[h] if h in test_results_macro else "" for h in
    #                                         metric_header])

    vwriter = csv.writer(valid_csv_f)
    vwriter.writerows(rows_valid)
    twriter = csv.writer(test_csv_f)
    twriter.writerows(rows_test)

    outf.close()
    valid_csv_f.close()
    test_csv_f.close()


if __name__ == "__main__":
    # hard coded
    calc_cl_metric = False
    BERTMODEL = "bert-base-uncased"  # TODO hard coded

    parser = argparse.ArgumentParser()
    # basic:
    parser.add_argument('--result_folder', '-r', type=str, default=None, help='result folder, to evaluate')
    parser.add_argument('--thresholds', type=int, nargs='+', default=None, help='user thresholds')
    # required which evaluation set we want to evaluate, random or genre ?
    parser.add_argument('--test_neg_strategy', type=str, default=None, help='negative sampling strategy')
    parser.add_argument('--valid_neg_strategy', type=str, default=None, help='negative sampling strategy')
    parser.add_argument('--best_epoch', type=str, default=None)

    # optional if we want to only calculate the metrics for users with certain review length.
    parser.add_argument('--user_review_len', type=int, default=None, help='min length of the user review')
    parser.add_argument('--review_field', type=str, default="review", help='review field')
    args, _ = parser.parse_known_args()

    result_folder = args.result_folder
    thrs = args.thresholds
    if thrs is None:
        thrs = []

    r_len = args.user_review_len
    r_field = args.review_field

    test_neg_strategy = args.test_neg_strategy
    valid_neg_strategy = args.valid_neg_strategy
    best_epoch = args.best_epoch

    if not os.path.exists(os.path.join(result_folder, "config.json")):
        raise ValueError(f"Result file config.json does not exist: {result_folder}")
    config = json.load(open(os.path.join(result_folder, "config.json")))

    start = time.time()
    test_prediction = {'predicted': {}}
    test_ground_truth = {'ground_truth': {}}
    test_user_refitem_items = defaultdict(lambda: defaultdict(set))
    valid_prediction = {'predicted': {}}
    valid_ground_truth = {'ground_truth': {}}
    valid_user_refitem_items = defaultdict(lambda: defaultdict(set))
    if test_neg_strategy is not None:
        test_prediction = json.load(open(os.path.join(result_folder,
                                                      f"test_predicted_test_neg_{test_neg_strategy}{f'_e-{best_epoch}' if best_epoch is not None else ''}.json")))
        test_ground_truth = json.load(open(os.path.join(result_folder,
                                                        f"test_ground_truth_test_neg_{test_neg_strategy}.json")))
        test_negs_with_refs = pd.read_csv(os.path.join(config['dataset']['dataset_path'], f"test_neg_{test_neg_strategy}.csv"), dtype=str)
        for user_id, item_id, ref_item_id in zip(test_negs_with_refs["user_id"], test_negs_with_refs["item_id"],
                                                 test_negs_with_refs["ref_item"]):
            test_user_refitem_items[user_id][ref_item_id].add(item_id)
        for user_id in test_user_refitem_items:
            for ref_item in test_user_refitem_items[user_id]:
                test_user_refitem_items[user_id][ref_item].add(ref_item)

    if valid_neg_strategy is not None:
        valid_prediction = json.load(open(os.path.join(result_folder,
                                                       f"best_valid_predicted_validation_neg_{valid_neg_strategy}{f'_e-{best_epoch}' if best_epoch is not None else ''}.json")))
        valid_ground_truth = json.load(open(os.path.join(result_folder,
                                                         f"best_valid_ground_truth_validation_neg_{valid_neg_strategy}.json")))
        valid_negs_with_refs = pd.read_csv(os.path.join(config['dataset']['dataset_path'], f"validation_neg_{test_neg_strategy}.csv"), dtype=str)
        for user_id, item_id, ref_item_id in zip(valid_negs_with_refs["user_id"], valid_negs_with_refs["item_id"],
                                                 valid_negs_with_refs["ref_item"]):
            valid_user_refitem_items[user_id][ref_item_id].add(item_id)
        for user_id in valid_user_refitem_items:
            for ref_item in valid_user_refitem_items[user_id]:
                valid_user_refitem_items[user_id][ref_item].add(ref_item)
    print(f"read data {time.time() - start}")

    # ranking_metrics = ["ndcg_cut_5", "ndcg_cut_10", "ndcg_cut_20", "P_1", "recip_rank"]
    ranking_metrics = ["ndcg_cut_5", "P_1", "recip_rank"]
    if (valid_neg_strategy is not None and "-" in valid_neg_strategy) or \
        (test_neg_strategy is not None and "-" in test_neg_strategy):
        ranking_metrics = ["ndcg_cut_5", "ndcg_cut_10", "ndcg_cut_20"]

    main(config, valid_ground_truth['ground_truth'], valid_prediction['predicted'],
         test_ground_truth['ground_truth'], test_prediction['predicted'],
         test_user_refitem_items, valid_user_refitem_items,
         thrs, r_len, r_field, test_neg_strategy, valid_neg_strategy, ranking_metrics)


# TODO test this script
