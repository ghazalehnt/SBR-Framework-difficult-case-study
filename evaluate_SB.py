import argparse
import json
import os
import time
from collections import defaultdict

import pandas as pd
import numpy as np

from SBR.utils.metrics import calculate_ranking_metrics_detailed

relevance_level = 1


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


def main(valid_gt, valid_pd, test_gt, test_pd, test_qid_items, valid_qid_items,
         test_neg_st=None, valid_neg_st=None, ranking_metrics=None):
    outfname = f"results_th_v-{valid_neg_st}_t-{test_neg_st}"
    if best_epoch is not None:
        outfname += f"_epoch-{best_epoch}"
    outfname += ".txt"
    print(outfname)
    outf = open(os.path.join(result_folder, outfname), 'w')

    # let's count how many interactions are there
    start = time.time()
    valid_pos_total_cnt = 0
    valid_neg_total_cnt = 0
    for u in valid_gt:
        valid_pos_total_cnt += len([k for k, v in test_gt[u].items() if v == 1])
        valid_neg_total_cnt += len([k for k, v in test_gt[u].items() if v == 0])

    test_pos_total_cnt = 0
    test_neg_total_cnt = 0
    for u in test_gt:
        test_pos_total_cnt += len([k for k, v in test_gt[u].items() if v == 1])
        test_neg_total_cnt += len([k for k, v in test_gt[u].items() if v == 0])
    print(f"count inters {time.time() - start}")

    outf.write(f"#total_evaluation_users = {len(set(test_gt.keys()).union(valid_gt.keys()))} \n")

    outf.write(f"#total_positive_inters_validation = {valid_pos_total_cnt}\n")
    outf.write(f"#total_negatove_inters_validation = {valid_neg_total_cnt}\n")

    outf.write(f"#total_positive_inters_test = {test_pos_total_cnt}\n")
    outf.write(f"#total_negatove_inters_test = {test_neg_total_cnt}\n")
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

    valid_results_micro, valid_results_macro = get_metrics(ground_truth=valid_gt_per_qid,
                                                           prediction_scores=valid_pd_per_qid,
                                                           weighted_labels=True if (valid_neg_st is not None and "-" in valid_neg_st) else False,
                                                           ranking_metrics=ranking_metrics)
    outf.write(f"MICRO Valid results ALL: {valid_results_micro}\n")
    outf.write(f"MACRO Valid results ALL: {valid_results_macro}\n")

    test_results_micro, test_results_macro = get_metrics(ground_truth=test_gt_per_qid,
                                                         prediction_scores=test_pd_per_qid,
                                                         weighted_labels=True if (test_neg_st is not None and "-" in test_neg_st) else False,
                                                         ranking_metrics=ranking_metrics)
    outf.write(f"MICRO Test results ALL: {test_results_micro}\n\n")
    outf.write(f"MACRO Test results ALL: {test_results_macro}\n\n")

    outf.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # basic:
    parser.add_argument('--result_folder', '-r', type=str, default=None, help='result folder, to evaluate')
    # required which evaluation set we want to evaluate, random or genre ?
    parser.add_argument('--test_neg_strategy', type=str, default=None, help='negative sampling strategy')
    parser.add_argument('--valid_neg_strategy', type=str, default=None, help='negative sampling strategy')
    parser.add_argument('--best_epoch', type=str, default=None)

    args, _ = parser.parse_known_args()

    result_folder = args.result_folder
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

    main(valid_ground_truth['ground_truth'], valid_prediction['predicted'],
         test_ground_truth['ground_truth'], test_prediction['predicted'],
         test_user_refitem_items, valid_user_refitem_items,
         test_neg_strategy, valid_neg_strategy, ranking_metrics)

