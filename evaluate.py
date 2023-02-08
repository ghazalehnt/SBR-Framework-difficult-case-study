import argparse
import json
import os
import time

from SBR.utils.metrics import calculate_ranking_metrics_macro_avg_over_qid

relevance_level = 1


def get_metrics(ground_truth, prediction_scores, weighted_labels, ranking_metrics=None):
    if len(ground_truth) == 0:
        return {}
    start = time.time()
    results = calculate_ranking_metrics_macro_avg_over_qid(gt=ground_truth, pd=prediction_scores,
                                                           relevance_level=relevance_level,
                                                           given_ranking_metrics=ranking_metrics,
                                                           calc_pytrec=not weighted_labels)
    print(f"ranking metrics in {time.time() - start}")
    return results


def main(valid_gt, valid_pd, test_gt, test_pd, test_neg_st=None, valid_neg_st=None, ranking_metrics=None):
    outfname = f"results_th_v-{valid_neg_st}_t-{test_neg_st}"
    if best_epoch is not None:
        outfname += f"_epoch-{best_epoch}"
    outfname += ".txt"
    print(outfname)
    outf = open(os.path.join(result_folder, outfname), 'w')

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
    #
    outf.write(f"#total_evaluation_users = {len(set(test_gt.keys()).union(valid_gt.keys()))} \n")
    #
    outf.write(f"#total_positive_inters_validation = {valid_pos_total_cnt}\n")
    outf.write(f"#total_negatove_inters_validation = {valid_neg_total_cnt}\n")
    #
    outf.write(f"#total_positive_inters_test = {test_pos_total_cnt}\n")
    outf.write(f"#total_negatove_inters_test = {test_neg_total_cnt}\n")
    outf.write("\n\n")

    # ALL:
    valid_results = get_metrics(ground_truth=valid_gt,
                                prediction_scores=valid_pd,
                                weighted_labels=True if (valid_neg_st is not None and "-" in valid_neg_st) else False,
                                ranking_metrics=ranking_metrics)
    outf.write(f"Valid results ALL: {valid_results}\n")

    test_results = get_metrics(ground_truth=test_gt,
                               prediction_scores=test_pd,
                               weighted_labels=True if (test_neg_st is not None and "-" in test_neg_st) else False,
                               ranking_metrics=ranking_metrics)
    outf.write(f"Test results ALL: {test_results}\n\n")

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

    start = time.time()
    test_prediction = {'predicted': {}}
    test_ground_truth = {'ground_truth': {}}
    valid_prediction = {'predicted': {}}
    valid_ground_truth = {'ground_truth': {}}
    if test_neg_strategy is not None:
        test_prediction = json.load(open(os.path.join(result_folder,
                                                      f"test_predicted_test_neg_{test_neg_strategy}{f'_e-{best_epoch}' if best_epoch is not None else ''}.json")))
        test_ground_truth = json.load(open(os.path.join(result_folder,
                                                        f"test_ground_truth_test_neg_{test_neg_strategy}.json")))
    if valid_neg_strategy is not None:
        valid_prediction = json.load(open(os.path.join(result_folder,
                                                       f"best_valid_predicted_validation_neg_{valid_neg_strategy}{f'_e-{best_epoch}' if best_epoch is not None else ''}.json")))
        valid_ground_truth = json.load(open(os.path.join(result_folder,
                                                         f"best_valid_ground_truth_validation_neg_{valid_neg_strategy}.json")))
    print(f"read data {time.time() - start}")

    # ranking_metrics = ["ndcg_cut_5", "ndcg_cut_10", "ndcg_cut_20", "P_1", "recip_rank"]
    ranking_metrics = ["ndcg_cut_5", "P_1", "recip_rank"]
    if (valid_neg_strategy is not None and "-" in valid_neg_strategy) or \
        (test_neg_strategy is not None and "-" in test_neg_strategy):
        ranking_metrics = ["ndcg_cut_5", "ndcg_cut_10", "ndcg_cut_20"]

    main(valid_ground_truth['ground_truth'], valid_prediction['predicted'],
         test_ground_truth['ground_truth'], test_prediction['predicted'],
         test_neg_strategy, valid_neg_strategy, ranking_metrics)

