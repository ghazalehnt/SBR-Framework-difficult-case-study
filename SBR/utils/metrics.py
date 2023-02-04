# how to do when micro averaging in cross validation ask andrew (do micro for each fold and then avg? or "concat" all results and do micro for all?)
import json
from collections import defaultdict

import pytrec_eval
from sklearn.metrics import ndcg_score
import numpy as np

from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD

ranking_metrics = [
    "ndcg_cut_5",
    "ndcg_cut_10",
    "ndcg_cut_20",
    "P_1",
    "recip_rank"
]

# TODO later remove the things with the weighted eval, such as using sklearn for ndcg... ??


def calculate_metrics(ground_truth, prediction_scores, users, items, relevance_level, given_ranking_metrics=None):
    # # qid= user1:{ item1:1 } ...
    gt = {str(u): {} for u in set(users)}
    pd = {str(u): {} for u in set(users)}
    # min_not_zero = 1
    for i in range(len(ground_truth)):
        if len(items) == 0:
            gt[str(users[i])][str(i)] = float(ground_truth[i])
            pd[str(users[i])][str(i)] = float(prediction_scores[i])
        else:
            gt[str(users[i])][str(items[i])] = float(ground_truth[i])
            pd[str(users[i])][str(items[i])] = float(prediction_scores[i])
        # if ground_truth[i] != 0 and ground_truth[i] < min_not_zero:
        #     min_not_zero = ground_truth[i]
    return calculate_ranking_metrics_macro_avg_over_qid(gt, pd, relevance_level, given_ranking_metrics, calc_pytrec=False)


def calculate_ranking_metrics_macro_avg_over_qid(gt, pd, relevance_level,
                                                   given_ranking_metrics=None, calc_pytrec=False):
    if given_ranking_metrics is None:
        given_ranking_metrics = ranking_metrics
    ndcg_metrics = [m for m in given_ranking_metrics if m.startswith("ndcg_")]
    results = calculate_ndcg(gt, pd, ndcg_metrics)
    if calc_pytrec:
        gt = {k: {k2: int(v2) for k2, v2 in v.items()} for k, v in gt.items()}
        r2 = calculate_ranking_metrics_pytreceval(gt, pd, relevance_level, given_ranking_metrics)
        for m, v in r2.items():
            if m in results:
                results[f"pytrec_{m}"] = v
            else:
                results[m] = v
    for m in results:
        assert len(results[m]) == len(gt)
        results[m] = np.array(results[m]).mean(axis=0).tolist()
    return results


def calculate_ranking_metrics_detailed(gt, pd, relevance_level,
                                       given_ranking_metrics=None, calc_pytrec=False):
    if given_ranking_metrics is None:
        given_ranking_metrics = ranking_metrics
    ndcg_metrics = [m for m in given_ranking_metrics if m.startswith("ndcg_")]
    results = calculate_ndcg(gt, pd, ndcg_metrics)
    if calc_pytrec:
        gt = {k: {k2: int(v2) for k2, v2 in v.items()} for k, v in gt.items()}
        r2 = calculate_ranking_metrics_pytreceval(gt, pd, relevance_level, given_ranking_metrics)
        for m, v in r2.items():
            if m in results:
                results[f"pytrec_{m}"] = v
            else:
                results[m] = v
    return results


def calculate_ranking_metrics_pytreceval(gt, pd, relevance_level, given_ranking_metrics):
    '''
    :param gt: dict of user -> item -> true score (relevance)
    :param pd: dict of user -> item -> predicted score
    :param relevance_level:
    :param given_ranking_metrics:
    :return: metric scores
    '''
    evaluator = pytrec_eval.RelevanceEvaluator(gt, given_ranking_metrics, relevance_level=int(relevance_level))
    scores = evaluator.evaluate(pd)
    per_qid_score = defaultdict()
    for m in given_ranking_metrics:
        per_qid_score[m] = [scores[qid][m] for qid in gt.keys()]
    # scores = [[metrics_dict.get(m, -1) for m in given_ranking_metrics] for metrics_dict in per_user_scores.values()]
    # scores = np.array(scores).mean(axis=0).tolist()
    # per_qid_scores = dict(zip(given_ranking_metrics, scores))
    return per_qid_score


def ndcg(gt, pd, k):
    per_qid_score = []
    for user in gt.keys():
        user_items = gt[user].keys()
        true_rel = [[gt[user][k] for k in user_items]]
        pred = [[pd[user][k] for k in user_items]]
        per_qid_score.append(ndcg_score(true_rel, pred, k=k))
    return per_qid_score


def calculate_ndcg(gt, pd, given_ranking_metrics):
    '''

    :param gt: dict of user -> item -> true score (relevance)
    :param pd: dict of user -> item -> predicted score
    :param relevance_level:
    :param given_ranking_metrics:
    :return: metric scores
    '''
    per_qid_score = defaultdict()
    for m in given_ranking_metrics:
        if m.startswith("ndcg_cut_"):
            per_qid_score[m] = ndcg(gt, pd, int(m[m.rindex("_")+1:]))
        else:
            raise NotImplementedError("other metrics not implemented")
    return per_qid_score


def log_results(ground_truth, prediction_scores, internal_user_ids, internal_items_ids,
                external_users, external_items, output_path_ground_truth, output_path_predicted, output_path_log=None):
    # we want to log the results corresponding to external user and item ids
    ex_users = external_users.to_pandas().set_index(INTERNAL_USER_ID_FIELD)
    user_ids = ex_users.loc[internal_user_ids].user_id.values
    ex_items = external_items.to_pandas().set_index(INTERNAL_ITEM_ID_FIELD)
    item_ids = ex_items.loc[internal_items_ids].item_id.values

    gt = {str(u): {} for u in sorted(set(user_ids))}
    pd = {str(u): {} for u in sorted(set(user_ids))}
    for i in range(len(ground_truth)):
        gt[str(user_ids[i])][str(item_ids[i])] = float(ground_truth[i])
        pd[str(user_ids[i])][str(item_ids[i])] = float(prediction_scores[i])
    json.dump({"predicted": pd}, open(output_path_predicted, 'w'))
    json.dump({"ground_truth": gt}, open(output_path_ground_truth, 'w'))
    cnt = 0
    if output_path_log is not None and 'text' in ex_users.columns:
        with open(output_path_log, "w") as f:
            for user_id in gt.keys():
                if cnt == 100:
                    break
                cnt += 1
                f.write(f"user:{user_id} - text:{ex_users[ex_users['user_id'] == user_id]['text'].values[0]}\n\n\n")
                for item_id, pd_score in sorted(pd[user_id].items(), key=lambda x:x[1], reverse=True):
                    f.write(f"item:{item_id} - label:{gt[user_id][item_id]} - score:{pd_score} - text:{ex_items[ex_items['item_id'] == item_id]['text'].values[0]}\n\n")
                f.write("-----------------------------\n")
