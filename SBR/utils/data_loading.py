import json
import pickle
import random
import time
from builtins import NotImplementedError
from collections import Counter, defaultdict
from os.path import join

import pandas as pd
import torch
import transformers
from datasets import Dataset, DatasetDict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import numpy as np

from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD
from SBR.utils.filter_user_profile import filter_user_profile_random_sentences


def tokenize_function(examples, tokenizer, field, max_length, max_num_chunks):
    result = tokenizer(
        examples[field],
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=True,
        padding="max_length"  # we pad the chunks here, because it would be too complicated later due to the chunks themselves...
    )

    sample_map = result.pop("overflow_to_sample_mapping")
    examples['chunks_input_ids'] = [[] for i in range(len(examples[field]))]
    examples['chunks_attention_mask'] = [[] for i in range(len(examples[field]))]
    for i, j in zip(sample_map, range(len(result['input_ids']))):
        if max_num_chunks is None or len(examples['chunks_input_ids'][i]) < max_num_chunks:
            examples['chunks_input_ids'][i].append(result['input_ids'][j])
            examples['chunks_attention_mask'][i].append(result['attention_mask'][j])
    return examples


def filter_user_profile(dataset_config, user_info):
    if dataset_config['user_text_filter'] == "random_sentence":
        user_info = filter_user_profile_random_sentences(dataset_config, user_info)
    else:
        raise ValueError(
            f"filtering method not implemented, or belong to another script! {dataset_config['user_text_filter']}")

    return Dataset.from_pandas(user_info, preserve_index=False)


def load_data(config, pretrained_model, for_precalc=False):
    start = time.time()
    print("Start: load dataset...")
    if 'user_text_filter' in config and config['user_text_filter'] in ["random_sentence"]:
        temp_cs = config['case_sensitive']
        config['case_sensitive'] = True
    datasets, user_info, item_info, filtered_out_user_item_pairs_by_limit = load_split_dataset(config, for_precalc)
    if 'user_text_filter' in config and config['user_text_filter'] in ["random_sentence"]:
        config['case_sensitive'] = temp_cs
    print(f"Finish: load dataset in {time.time()-start}")

    # apply filter:
    if 'text' in user_info.column_names and config['user_text_filter'] != "":
        user_info = filter_user_profile(config, user_info)

    # tokenize when needed:
    return_padding_token = None
    padding_token = None
    if pretrained_model is not None and config["load_tokenized_text_in_batch"] is True:
        tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model)
        padding_token = tokenizer.pad_token_id
        return_padding_token = tokenizer.pad_token_id
        if 'text' in user_info.column_names:
            user_info = user_info.map(tokenize_function, batched=True,
                                      fn_kwargs={"tokenizer": tokenizer, "field": 'text',
                                                 # this is used to know how big should the chunks be, because the model may have extra stuff to add to the chunks
                                                 "max_length": config["user_chunk_size"],
                                                 "max_num_chunks": config['max_num_chunks_user'] if "max_num_chunks_user" in config else None
                                                 })
            user_info = user_info.remove_columns(['text'])
        if 'text' in item_info.column_names:
            item_info = item_info.map(tokenize_function, batched=True,
                                      fn_kwargs={"tokenizer": tokenizer, "field": 'text',
                                                 # this is used to know how big should the chunks be, because the model may have extra stuff to add to the chunks
                                                 "max_length": config["item_chunk_size"],
                                                 "max_num_chunks": config['max_num_chunks_item'] if "max_num_chunks_item" in config else None
                                                 })
            item_info = item_info.remove_columns(['text'])

    train_dataloader, validation_dataloader, test_dataloader = None, None, None
    if not for_precalc:
        start = time.time()
        print("Start: get user used items...")
        user_used_items = get_user_used_items(datasets, filtered_out_user_item_pairs_by_limit)
        print(f"Finish: get user used items in {time.time() - start}")

        # when we need text for the training. we sort of check it if the passed padding_token is not none in collate_fns, so this is set here now:
        if config['load_tokenized_text_in_batch'] is False:
            padding_token = None  # this causes the collate functions to

        train_collate_fn = None
        valid_collate_fn = None
        test_collate_fn = None
        start = time.time()
        if 'training_neg_sampling_strategy' in config:
            if config['training_neg_sampling_strategy'] == "random":
                print("Start: train collate_fn initialize...")
                cur_used_items = user_used_items['train'].copy()
                train_collate_fn = CollateNegSamplesRandomOpt(config['training_neg_samples'],
                                                              cur_used_items, user_info,
                                                              item_info, padding_token=padding_token)
                print(f"Finish: train collate_fn initialize {time.time() - start}")
            elif config['training_neg_sampling_strategy'].startswith("random_w_CF_dot_"):
                cur_used_items = user_used_items['train'].copy()
                label_weight_name = config['training_neg_sampling_strategy']
                oldmax = int(label_weight_name[label_weight_name.rindex("_") + 1:])
                oldmin = int(label_weight_name[
                             label_weight_name.index("w_CF_dot_") + len("w_CF_dot_"):label_weight_name.rindex("_")])
                train_collate_fn = CollateNegSamplesRandomCFWeighted(config['training_neg_samples'],
                                                                     cur_used_items,
                                                                     user_used_items['train'],
                                                                     config["cf_sim_checkpoint"],
                                                                     config["cf_internal_ids"],
                                                                     oldmax,
                                                                     oldmin,
                                                                     user_info,
                                                                     item_info, padding_token=padding_token)
                print(f"Finish: train collate_fn initialize {time.time() - start}")
            elif config['training_neg_sampling_strategy'] == "":
                train_collate_fn = CollateOriginalDataPad(user_info, item_info, padding_token)
            elif config['training_neg_sampling_strategy'] == "genres":
                print("Start: train collate_fn initialize...")
                cur_used_items = user_used_items['train'].copy()
                train_collate_fn = CollateNegSamplesGenresOpt(config['training_neg_sampling_strategy'],
                                                              config['training_neg_samples'], cur_used_items, user_info,
                                                              item_info, padding_token=padding_token)
                print(f"Finish: train collate_fn initialize {time.time() - start}")

            train_dataloader = DataLoader(datasets['train'],
                                          batch_size=config['train_batch_size'],
                                          shuffle=True,
                                          collate_fn=train_collate_fn,
                                          num_workers=config['dataloader_num_workers']
                                          )

        if 'validation_neg_sampling_strategy' in config:
            if config['validation_neg_sampling_strategy'] == "random":
                start = time.time()
                print("Start: used_item copy and validation collate_fn initialize...")
                cur_used_items = user_used_items['train'].copy()
                for user_id, u_items in user_used_items['validation'].items():
                    cur_used_items[user_id] = cur_used_items[user_id].union(u_items)
                valid_collate_fn = CollateNegSamplesRandomOpt(config['validation_neg_samples'], cur_used_items,
                                                              padding_token=padding_token)
                print(f"Finish: used_item copy and validation collate_fn initialize {time.time() - start}")
            elif config['validation_neg_sampling_strategy'].startswith("f:"):
                start = time.time()
                print("Start: used_item copy and validation collate_fn initialize...")
                valid_collate_fn = CollateOriginalDataPad(user_info, item_info, padding_token=padding_token)
                print(f"Finish: used_item copy and validation collate_fn initialize {time.time() - start}")

            validation_dataloader = DataLoader(datasets['validation'],
                                               batch_size=config['eval_batch_size'],
                                               collate_fn=valid_collate_fn,
                                               num_workers=config['dataloader_num_workers'])

        if 'test_neg_sampling_strategy' in config:
            if config['test_neg_sampling_strategy'] == "random":
                start = time.time()
                print("Start: used_item copy and test collate_fn initialize...")
                cur_used_items = user_used_items['train'].copy()
                for user_id, u_items in user_used_items['validation'].items():
                    cur_used_items[user_id] = cur_used_items[user_id].union(u_items)
                for user_id, u_items in user_used_items['test'].items():
                    cur_used_items[user_id] = cur_used_items[user_id].union(u_items)
                test_collate_fn = CollateNegSamplesRandomOpt(config['test_neg_samples'], cur_used_items,
                                                             padding_token=padding_token)
                print(f"Finish: used_item copy and test collate_fn initialize {time.time() - start}")
            elif config['test_neg_sampling_strategy'].startswith("f:"):
                start = time.time()
                print("Start: used_item copy and test collate_fn initialize...")
                test_collate_fn = CollateOriginalDataPad(user_info, item_info, padding_token=padding_token)
                print(f"Finish: used_item copy and test collate_fn initialize {time.time() - start}")

            test_dataloader = DataLoader(datasets['test'],
                                         batch_size=config['eval_batch_size'],
                                         collate_fn=test_collate_fn,
                                         num_workers=config['dataloader_num_workers'])

    return train_dataloader, validation_dataloader, test_dataloader, user_info, item_info, config['relevance_level'], return_padding_token


class CollateOriginalDataPad(object):
    def __init__(self, user_info, item_info, padding_token=None):
        self.user_info = user_info.to_pandas()
        self.item_info = item_info.to_pandas()
        self.padding_token = padding_token

    def __call__(self, batch):
        batch_df = pd.DataFrame(batch)
        if self.padding_token is not None:
            # user:
            temp_user = self.user_info.loc[batch_df[INTERNAL_USER_ID_FIELD]][['chunks_input_ids', 'chunks_attention_mask']]\
                .reset_index().drop(columns=['index'])
            temp_user = pd.concat([batch_df, temp_user], axis=1)
            temp_user = temp_user.rename(columns={"chunks_input_ids": "user_chunks_input_ids",
                                                  "chunks_attention_mask": "user_chunks_attention_mask"})
            # item:
            temp_item = self.item_info.loc[batch_df[INTERNAL_ITEM_ID_FIELD]][['chunks_input_ids', 'chunks_attention_mask']] \
                .reset_index().drop(columns=['index'])
            temp_item = pd.concat([batch_df, temp_item], axis=1)
            temp_item = temp_item.rename(columns={"chunks_input_ids": "item_chunks_input_ids",
                                                  "chunks_attention_mask": "item_chunks_attention_mask"})
            temp = pd.merge(temp_user, temp_item, on=['label', 'internal_user_id', 'internal_item_id'])

            # pad ,  the resulting tensor is num-chunks * batch * tokens -> bcs later we want to do batchwise
            ret = {}
            for col in ["user_chunks_input_ids", "user_chunks_attention_mask", "item_chunks_input_ids", "item_chunks_attention_mask"]:
                instances = [torch.tensor([list(t) for t in instance]) for instance in temp[col]]
                ret[col] = pad_sequence(instances, padding_value=self.padding_token).type(torch.int64)
            for col in temp.columns:
                if col in ret:
                    continue
                ret[col] = torch.tensor(temp[col]).unsqueeze(1)
        else:
            ret = {}
            for col in batch_df.columns:
                if col in ret:
                    continue
                ret[col] = torch.tensor(batch_df[col]).unsqueeze(1)
        return ret


class CollateRepresentationBuilder(object):
    def __init__(self, padding_token):
        self.padding_token = padding_token

    def __call__(self, batch):
        batch_df = pd.DataFrame(batch)
        ret = {}
        for col in ["chunks_input_ids", "chunks_attention_mask"]:
            instances = [torch.tensor([list(t) for t in instance]) for instance in batch_df[col]]
            ret[col] = pad_sequence(instances, padding_value=self.padding_token).type(torch.int64)
        for col in batch_df.columns:
            if col in ret:
                continue
            if col in ["user_id", "item_id"]:
                ret[col] = batch_df[col]
            else:
                ret[col] = torch.tensor(batch_df[col]).unsqueeze(1)
        return ret


class CollateNegSamplesRandomOpt(object):
    def __init__(self, num_neg_samples, used_items, user_info=None, item_info=None, padding_token=None):
        self.num_neg_samples = num_neg_samples
        self.used_items = used_items
        # pool of all items is created from seen training items:
        all_items = []
        for items in self.used_items.values():
            all_items.extend(items)
        self.all_items = list(set(all_items))
        self.user_info = user_info.to_pandas()
        self.item_info = item_info.to_pandas()
        self.padding_token = padding_token

    def sample(self, batch_df):
        user_counter = Counter(batch_df[INTERNAL_USER_ID_FIELD])
        samples = []
        for user_id in user_counter.keys():
            num_pos = user_counter[user_id]
            max_num_user_neg_samples = min(len(self.all_items), num_pos * self.num_neg_samples)
            if max_num_user_neg_samples < num_pos * self.num_neg_samples:
                print(f"WARN: user {user_id} needed {num_pos * self.num_neg_samples} samples,"
                      f"but all_items are {len(self.all_items)}")
                pass
            user_samples = set()
            try_cnt = -1
            num_user_neg_samples = max_num_user_neg_samples
            while True:
                if try_cnt == 100:
                    print(f"WARN: After {try_cnt} tries, could not find {max_num_user_neg_samples} samples for"
                          f"{user_id}. We instead have {len(user_samples)} samples.")
                    break
                current_samples = set(random.sample(self.all_items, num_user_neg_samples))
                current_samples -= user_samples
                cur_used_samples = self.used_items[user_id].intersection(current_samples)
                current_samples = current_samples - cur_used_samples
                user_samples = user_samples.union(current_samples)
                num_user_neg_samples = max(max_num_user_neg_samples - len(user_samples), 0)
                if len(user_samples) < max_num_user_neg_samples:
                    # to make the process faster
                    if num_user_neg_samples < len(user_samples):
                        num_user_neg_samples = min(max_num_user_neg_samples, num_user_neg_samples * 2)
                    try_cnt += 1
                else:
                    if len(user_samples) > max_num_user_neg_samples:
                        user_samples = set(list(user_samples)[:max_num_user_neg_samples])
                    break
            samples.extend([{'label': 0, INTERNAL_USER_ID_FIELD: user_id, INTERNAL_ITEM_ID_FIELD: sampled_item_id}
                            for sampled_item_id in user_samples])
        return samples

    def __call__(self, batch):
        batch_df = pd.DataFrame(batch)
        samples = self.sample(batch_df)
        batch_df = pd.concat([batch_df, pd.DataFrame(samples)]).reset_index().drop(columns=['index'])

        if self.padding_token is not None:
            # user:
            temp_user = self.user_info.loc[batch_df[INTERNAL_USER_ID_FIELD]][['chunks_input_ids', 'chunks_attention_mask']] \
                .reset_index().drop(columns=['index'])
            temp_user = pd.concat([batch_df, temp_user], axis=1)
            temp_user = temp_user.rename(columns={"chunks_input_ids": "user_chunks_input_ids",
                                                  "chunks_attention_mask": "user_chunks_attention_mask"})
            # item:
            temp_item = self.item_info.loc[batch_df[INTERNAL_ITEM_ID_FIELD]][['chunks_input_ids', 'chunks_attention_mask']] \
                .reset_index().drop(columns=['index'])
            temp_item = pd.concat([batch_df, temp_item], axis=1)
            temp_item = temp_item.rename(columns={"chunks_input_ids": "item_chunks_input_ids",
                                                  "chunks_attention_mask": "item_chunks_attention_mask"})
            temp = pd.merge(temp_user, temp_item, on=['label', 'internal_user_id', 'internal_item_id'])

            # pad ,  the resulting tensor is num-chunks * batch * tokens -> bcs later we want to do batchwise
            ret = {}
            for col in ["user_chunks_input_ids", "user_chunks_attention_mask", "item_chunks_input_ids",
                        "item_chunks_attention_mask"]:
                instances = [torch.tensor([list(t) for t in instance]) for instance in temp[col]]
                ret[col] = pad_sequence(instances, padding_value=self.padding_token).type(torch.int64)
            for col in temp.columns:
                if col in ret:
                    continue
                ret[col] = torch.tensor(temp[col]).unsqueeze(1)
        else:
            ret = {}
            for col in batch_df.columns:
                if col in ret:
                    continue
                ret[col] = torch.tensor(batch_df[col]).unsqueeze(1)
        return ret


class CollateNegSamplesRandomCFWeighted(CollateNegSamplesRandomOpt):
    def __init__(self, num_neg_samples, used_items, user_training_items,
                 cf_checkpoint_file, cf_item_id_file, oldmax, oldmin,
                 user_info=None, item_info=None, padding_token=None):
        self.num_neg_samples = num_neg_samples
        self.used_items = used_items
        # pool of all items is created from seen training items:
        all_items = []
        for items in self.used_items.values():
            all_items.extend(items)
        self.all_items = list(set(all_items))
        self.padding_token = padding_token
        self.user_training_items = user_training_items
        temp = torch.load(cf_checkpoint_file, map_location=torch.device('cpu'))['model_state_dict']['item_embedding.weight']
        cf_item_internal_ids = json.load(open(cf_item_id_file, 'r'))
        item_info = item_info.to_pandas()
        self.cf_item_reps = {item_in: temp[cf_item_internal_ids[item_ex]] for item_ex, item_in in zip(item_info["item_id"], item_info[INTERNAL_ITEM_ID_FIELD])}
        self.oldmax = oldmax
        self.oldmin = oldmin
        if self.padding_token is not None:
            self.user_info = user_info.to_pandas()
            self.item_info = item_info

    def sample(self, batch_df):
        user_counter = Counter(batch_df[INTERNAL_USER_ID_FIELD])
        samples = []
        for user_id in user_counter.keys():
            num_pos = user_counter[user_id]
            max_num_user_neg_samples = min(len(self.all_items), num_pos * self.num_neg_samples)
            if max_num_user_neg_samples < num_pos * self.num_neg_samples:
                print(f"WARN: user {user_id} needed {num_pos * self.num_neg_samples} samples,"
                      f"but all_items are {len(self.all_items)}")
                pass
            user_samples = set()
            try_cnt = -1
            num_user_neg_samples = max_num_user_neg_samples
            while True:
                if try_cnt == 100:
                    print(f"WARN: After {try_cnt} tries, could not find {max_num_user_neg_samples} samples for"
                          f"{user_id}. We instead have {len(user_samples)} samples.")
                    break
                current_samples = set(random.sample(self.all_items, num_user_neg_samples))
                current_samples -= user_samples
                cur_used_samples = self.used_items[user_id].intersection(current_samples)
                current_samples = current_samples - cur_used_samples
                user_samples = user_samples.union(current_samples)
                num_user_neg_samples = max(max_num_user_neg_samples - len(user_samples), 0)
                if len(user_samples) < max_num_user_neg_samples:
                    # to make the process faster
                    if num_user_neg_samples < len(user_samples):
                        num_user_neg_samples = min(max_num_user_neg_samples, num_user_neg_samples * 2)
                    try_cnt += 1
                else:
                    if len(user_samples) > max_num_user_neg_samples:
                        user_samples = set(list(user_samples)[:max_num_user_neg_samples])
                    break
            # now we calculate the label weight and add the unlabeled samples to the samples list
            for sampled_item_id in user_samples:
                sims = []
                for pos_item in self.user_training_items[user_id]:
                    s = np.dot(self.cf_item_reps[sampled_item_id], self.cf_item_reps[pos_item])
                    s = (s - self.oldmin) / (self.oldmax - self.oldmin)  # s = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
                    # as oldmin and oldmax are estimates, we make sure that the s is between 0 and 1:
                    s = max(0, s)
                    s = min(1, s)
                    sims.append(s)
                avg_sim = sum(sims) / len(sims)
                samples.append({'label': avg_sim,
                                INTERNAL_USER_ID_FIELD: user_id,
                                INTERNAL_ITEM_ID_FIELD: sampled_item_id})
        return samples


class CollateNegSamplesGenresOpt(CollateNegSamplesRandomOpt):
    def __init__(self, strategy, num_neg_samples, used_items, user_info=None, item_info=None, padding_token=None):
        self.num_neg_samples = num_neg_samples
        self.used_items = used_items
        # pool of all items is created from seen training items:
        self.user_info = user_info.to_pandas()
        self.item_info = item_info.to_pandas()
        genres_field = 'category' if 'category' in self.item_info.columns else 'genres'
        print("start parsing item genres")
        self.genre_items = defaultdict(set)
        self.item_genres = defaultdict(list)
        for item, genres in zip(self.item_info[INTERNAL_ITEM_ID_FIELD], self.item_info[genres_field]):
            for g in [g.replace("'", "").replace('"', "").replace("[", "").replace("]", "").strip() for g in genres.split(",")]:
                self.genre_items[g].add(item)
                self.item_genres[item].append(g)
        for g in self.genre_items:
            self.genre_items[g] = list(self.genre_items[g])
        print("finish parsing item genres")

        self.strategy = strategy
        self.padding_token = padding_token

    def sample(self, batch_df):
        samples = []
        for user_id, item_id in zip(batch_df[INTERNAL_USER_ID_FIELD], batch_df[INTERNAL_ITEM_ID_FIELD]):
            sampled_genres = []
            while len(sampled_genres) < self.num_neg_samples:
                sampled_genres.extend(np.random.choice(self.item_genres[item_id],
                                                       min(self.num_neg_samples - len(sampled_genres),
                                                           len(self.item_genres[item_id])),
                                                       replace=False).tolist())
            neg_samples = set()
            for g in sampled_genres:
                try_cnt = -1
                while True:
                    if try_cnt == 20:
                        break
                    s = random.choice(self.genre_items[g])
                    if s not in neg_samples and s not in self.used_items[user_id]:
                        neg_samples.add(s)
                        break
                    try_cnt += 1
            samples.extend([{'label': 0, INTERNAL_USER_ID_FIELD: user_id, INTERNAL_ITEM_ID_FIELD: sampled_item_id}
                            for sampled_item_id in neg_samples])
        return samples


def get_user_used_items(datasets, filtered_out_user_item_pairs_by_limit):
    used_items = {}
    for split in datasets.keys():
        used_items[split] = defaultdict(set)
        for user_iid, item_iid in zip(datasets[split][INTERNAL_USER_ID_FIELD], datasets[split][INTERNAL_ITEM_ID_FIELD]):
            used_items[split][user_iid].add(item_iid)

    for user, books in filtered_out_user_item_pairs_by_limit.items():
        used_items['train'][user] = used_items['train'][user].union(books)

    return used_items


def load_split_dataset(config, for_precalc=False):
    user_text_fields = config['user_text']
    item_text_fields = config['item_text']
    if config['load_user_item_text'] is False:
        user_text_fields = []
        item_text_fields = []

    # read users and items, create internal ids for them to be used
    keep_fields = ["user_id"]
    keep_fields.extend([field[field.index("user.")+len("user."):] for field in user_text_fields if "user." in field])
    keep_fields.extend([field[field.index("user.")+len("user."):] for field in item_text_fields if "user." in field])
    keep_fields = list(set(keep_fields))
    user_info = pd.read_csv(join(config['dataset_path'], "users.csv"), usecols=keep_fields, dtype=str)
    user_info = user_info.sort_values("user_id").reset_index(drop=True) # this is crucial, as the precomputing is done with internal ids
    user_info[INTERNAL_USER_ID_FIELD] = np.arange(0, user_info.shape[0])
    user_info = user_info.fillna('')
    user_info = user_info.rename(
        columns={field[field.index("user.") + len("user."):]: field for field in user_text_fields if
                 "user." in field})

    keep_fields = ["item_id"]
    keep_fields.extend([field[field.index("item.")+len("item."):] for field in item_text_fields if "item." in field])
    keep_fields.extend([field[field.index("item.") + len("item."):] for field in user_text_fields if "item." in field])
    keep_fields = list(set(keep_fields))
    tie_breaker = None
    if len(user_text_fields) > 0 and 'user_item_text_tie_breaker' in config and config['user_item_text_tie_breaker'] != "":
        if config['user_item_text_tie_breaker'].startswith("item."):
            tie_breaker = config['user_item_text_tie_breaker']
            tie_breaker = tie_breaker[tie_breaker.index("item.") + len("item."):]
            keep_fields.extend([tie_breaker])
        else:
            raise ValueError(f"tie-breaker value: {config['user_item_text_tie_breaker']}")

    if not for_precalc and config["training_neg_sampling_strategy"] == "genres":
        if config["name"] == "Amazon":
            keep_fields.append("category")
        elif config["name"] in ["CGR", "GR_UCSD"]:
            keep_fields.append("genres")
        else:
            raise NotImplementedError()
    item_info = pd.read_csv(join(config['dataset_path'], "items.csv"), usecols=keep_fields, low_memory=False, dtype=str)
    if tie_breaker is not None:
        if tie_breaker in ["avg_rating", "average_rating"]:
            item_info[tie_breaker] = item_info[tie_breaker].astype(float)
        else:
            raise NotImplementedError(f"tie-break {tie_breaker} not implemented")
        item_info[tie_breaker] = item_info[tie_breaker].fillna(0)
    item_info = item_info.sort_values("item_id").reset_index(drop=True)  # this is crucial, as the precomputing is done with internal ids
    item_info[INTERNAL_ITEM_ID_FIELD] = np.arange(0, item_info.shape[0])
    item_info = item_info.fillna('')
    item_info = item_info.rename(
        columns={field[field.index("item.") + len("item."):]: field for field in item_text_fields if
                 "item." in field})
    for field in []:
        if field in item_info.columns:
            item_info[field] = item_info[field].apply(lambda x: ", ".join(x.split(",")).replace("  ", " ").strip())

    # read user-item interactions, map the user and item ids to the internal ones
    sp_files = {"train": join(config['dataset_path'], "train.csv"),
                "validation": join(config['dataset_path'], "validation.csv"),
                "test": join(config['dataset_path'], "test.csv")}
    split_datasets = {}
    filtered_out_user_item_pairs_by_limit = defaultdict(set)
    for sp, file in sp_files.items():
        df = pd.read_csv(file, dtype=str)  # rating:float64
        # if predicting rating: remove the not-rated entries and map rating text to int
        df = df[df['rating'].notna()].reset_index()
        df['rating'] = df['rating'].astype(int)
        df['label'] = df['rating']
        df['rating'] = df['rating'].fillna(-1)

        # replace user_id with internal_user_id (same for item_id)
        df = df.merge(user_info[["user_id", INTERNAL_USER_ID_FIELD]], "left", on="user_id")
        df = df.merge(item_info[["item_id", INTERNAL_ITEM_ID_FIELD]], "left", on="item_id")
        df = df.drop(columns=["user_id", "item_id"])

        df = df.rename(
            columns={field[field.index("interaction.") + len("interaction."):]: field for field in user_text_fields if
                     "interaction." in field})
        df = df.rename(
            columns={field[field.index("interaction.") + len("interaction."):]: field for field in item_text_fields if
                     "interaction." in field})

        for field in user_text_fields:
            if "interaction." in field:
                df[field] = df[field].fillna('')

        # concat and move the user/item text fields to user and item info:
        sort_reviews = None
        if len(user_text_fields) > 0:
            if 'user_item_text_choice' in config and len(config['user_item_text_choice']) > 0:
                sort_reviews = config['user_item_text_choice']
        # text profile:
        if sp == 'train':
            ## USER:
            # This code works for user text fields from interaction and item file
            user_item_text_fields = [field for field in user_text_fields if "item." in field]
            user_inter_text_fields = [field for field in user_text_fields if "interaction." in field]
            user_item_inter_text_fields = user_item_text_fields.copy()
            user_item_inter_text_fields.extend(user_inter_text_fields)

            if len(user_item_inter_text_fields) > 0:
                user_item_merge_fields = [INTERNAL_ITEM_ID_FIELD]
                user_item_merge_fields.extend(user_item_text_fields)
                if tie_breaker in ["avg_rating", "average_rating"]:
                    user_item_merge_fields.append(tie_breaker)

                user_inter_merge_fields = [INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD, 'rating']
                user_inter_merge_fields.extend(user_inter_text_fields)

                temp = df[user_inter_merge_fields].\
                    merge(item_info[user_item_merge_fields], on=INTERNAL_ITEM_ID_FIELD)
                if sort_reviews.startswith("pos_rating_sorted_"):
                    pos_threshold = int(sort_reviews[sort_reviews.rindex("_") + 1:])
                    temp = temp[temp['rating'] >= pos_threshold]
                # before sorting them based on rating, etc., let's append each row's field together (e.g. title. genres. review.)
                temp['text'] = temp[user_item_inter_text_fields].agg('. '.join, axis=1)

                if sort_reviews is None:
                    temp = temp.groupby(INTERNAL_USER_ID_FIELD)['text']
                else:
                    if sort_reviews == "rating_sorted" or sort_reviews.startswith("pos_rating_sorted_"):
                        if tie_breaker is None:
                            temp = temp.sort_values('rating', ascending=False).groupby(
                                INTERNAL_USER_ID_FIELD)['text']
                        elif tie_breaker in ["avg_rating", "average_rating"]:
                            temp = temp.sort_values(['rating', tie_breaker], ascending=[False, False]).groupby(
                                INTERNAL_USER_ID_FIELD)['text']
                        else:
                            raise ValueError("Not implemented!")
                    else:
                        raise ValueError("Not implemented!")
                temp = temp.apply('. '.join).reset_index()
                user_info = user_info.merge(temp, "left", on=INTERNAL_USER_ID_FIELD)
                user_info['text'] = user_info['text'].fillna('')

            ## ITEM:
            # This code works for item text fields from interaction and user file
            item_user_text_fields = [field for field in item_text_fields if "user." in field]
            item_inter_text_fields = [field for field in item_text_fields if "interaction." in field]
            item_user_inter_text_fields = item_user_text_fields.copy()
            item_user_inter_text_fields.extend(item_inter_text_fields)

            if len(item_user_inter_text_fields) > 0:
                item_user_merge_fields = [INTERNAL_USER_ID_FIELD]
                item_user_merge_fields.extend(item_user_text_fields)

                item_inter_merge_fields = [INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD, 'rating']
                item_inter_merge_fields.extend(item_inter_text_fields)

                temp = df[item_inter_merge_fields]. \
                    merge(user_info[item_user_merge_fields], on=INTERNAL_USER_ID_FIELD)
                if sort_reviews.startswith("pos_rating_sorted_"):
                    pos_threshold = int(sort_reviews[sort_reviews.rindex("_") + 1:])
                    temp = temp[temp['rating'] >= pos_threshold]
                # before sorting them based on rating, etc., let's append each row's field together
                temp['text'] = temp[item_user_inter_text_fields].agg('. '.join, axis=1)

                if sort_reviews is None:
                    temp = temp.groupby(INTERNAL_ITEM_ID_FIELD)['text'].apply('. '.join).reset_index()
                else:
                    if sort_reviews == "rating_sorted" or sort_reviews.startswith("pos_rating_sorted_"):
                        temp = temp.sort_values('rating', ascending=False).groupby(
                            INTERNAL_ITEM_ID_FIELD)['text'].apply('. '.join).reset_index()
                    else:
                        raise ValueError("Not implemented!")

                item_info = item_info.merge(temp, "left", on=INTERNAL_ITEM_ID_FIELD)
                item_info['text'] = item_info['text'].fillna('')

        # remove the rest
        remove_fields = df.columns
        print(f"interaction fields: {remove_fields}")
        keep_fields = ["label", INTERNAL_ITEM_ID_FIELD, INTERNAL_USER_ID_FIELD]
        remove_fields = list(set(remove_fields) - set(keep_fields))
        df = df.drop(columns=remove_fields)
        split_datasets[sp] = df

    # after moving text fields to user/item info, now concatenate them all and create a single 'text' field:
    user_remaining_text_fields = [field for field in user_text_fields if field.startswith("user.")]
    if 'text' in user_info.columns:
        user_remaining_text_fields.append('text')
    if len(user_remaining_text_fields) > 0:
        user_info['text'] = user_info[user_remaining_text_fields].agg('. '.join, axis=1)
        if not config['case_sensitive']:
            user_info['text'] = user_info['text'].apply(str.lower)
        if config['normalize_negation']:
            user_info['text'] = user_info['text'].replace("n\'t", " not", regex=True)
        user_info = user_info.drop(columns=[field for field in user_text_fields if field.startswith("user.")])

    item_remaining_text_fields = [field for field in item_text_fields if field.startswith("item.")]
    if 'text' in item_info.columns:
        item_remaining_text_fields.append('text')
    if len(item_remaining_text_fields) > 0:
        item_info['text'] = item_info[item_remaining_text_fields].agg('. '.join, axis=1)
        if not config['case_sensitive']:
            item_info['text'] = item_info['text'].apply(str.lower)
        if config['normalize_negation']:
            item_info['text'] = item_info['text'].replace("n\'t", " not", regex=True)
    if config["training_neg_sampling_strategy"] == "genres":
        if config["name"] == "Amazon":
            if 'category' not in item_info.columns:
                item_info['category'] = item_info['item.category']
        elif config["name"] in ["CGR", "GR_UCSD"]:
            if 'genres' not in item_info.columns:
                item_info['genres'] = item_info['item.genres']
        else:
            raise NotImplementedError()
    item_info = item_info.drop(columns=[field for field in item_text_fields if field.startswith("item.")])

    # loading negative samples for eval sets: I used to load them in a collatefn, but, because batch=101 does not work for evaluation for BERT-based models
    user_item_sim = None
    if not for_precalc and 'validation_neg_sampling_strategy' in config and\
            config['validation_neg_sampling_strategy'].startswith("f:"):
        label_weight = 0
        if "-" in config['validation_neg_sampling_strategy']:
            fname = config['validation_neg_sampling_strategy'][2:config['validation_neg_sampling_strategy'].index("-")]
            label_weight_name = config['validation_neg_sampling_strategy'][config['validation_neg_sampling_strategy'].index("-")+1:]
            if label_weight_name.startswith("w_CF_dot_"):
                label_weight = None
                oldmax = int(label_weight_name[label_weight_name.rindex("_")+1:])
                oldmin = int(label_weight_name[label_weight_name.index("w_CF_dot_")+len("w_CF_dot_"):label_weight_name.rindex("_")])
                sim_st = "dot"
                user_item_sim = pickle.load(open(join(config['dataset_path'], f"eval_user_item_CF_sim_{sim_st}_scaled_{oldmin}-{oldmax}.pkl"), 'rb'))
            else:
                raise NotImplementedError(f"{label_weight} not implemented")
        else:
            fname = config['validation_neg_sampling_strategy'][2:]
        negs = pd.read_csv(join(config['dataset_path'], fname+".csv"), dtype=str)
        if label_weight is not None:
            negs['label'] = label_weight
        else:
            labels = []
            if label_weight_name.startswith("w_CF_dot_"):
                for user, unlabeled_item in zip(negs['user_id'], negs['item_id']):
                    labels.append(user_item_sim[user][unlabeled_item])
            negs['label'] = labels
        negs = negs.merge(user_info[["user_id", INTERNAL_USER_ID_FIELD]], "left", on="user_id")
        negs = negs.merge(item_info[["item_id", INTERNAL_ITEM_ID_FIELD]], "left", on="item_id")
        negs = negs.drop(columns=["user_id", "item_id"])
        if "ref_item" in negs.columns:
            negs = negs.drop(columns=["ref_item"])
        split_datasets['validation'] = pd.concat([split_datasets['validation'], negs])
        split_datasets['validation'] = split_datasets['validation'].sort_values(INTERNAL_USER_ID_FIELD).reset_index().drop(columns=['index'])

    if not for_precalc and 'test_neg_sampling_strategy' in config and \
            config['test_neg_sampling_strategy'].startswith("f:"):
        label_weight = 0
        if "-" in config['test_neg_sampling_strategy']:
            fname = config['test_neg_sampling_strategy'][2:config['test_neg_sampling_strategy'].index("-")]
            label_weight_name = config['test_neg_sampling_strategy'][
                                config['test_neg_sampling_strategy'].index("-") + 1:]
            if label_weight_name.startswith("w_CF_dot"):
                label_weight = None
                oldmax = int(label_weight_name[label_weight_name.rindex("_")+1:])
                oldmin = int(label_weight_name[label_weight_name.index("w_CF_dot_")+len("w_CF_dot_"):label_weight_name.rindex("_")])
                sim_st = "dot"
                if user_item_sim is None:
                    user_item_sim = pickle.load(open(
                        join(config['dataset_path'], f"eval_user_item_CF_sim_{sim_st}_scaled_{oldmin}-{oldmax}.pkl"), 'rb'))
            else:
                raise NotImplementedError(f"{label_weight} not implemented")
        else:
            fname = config['test_neg_sampling_strategy'][2:]
        negs = pd.read_csv(join(config['dataset_path'], fname+".csv"), dtype=str)
        if label_weight is not None:
            negs['label'] = label_weight
        else:
            labels = []
            if label_weight_name.startswith("w_CF_dot_"):
                for user, unlabeled_item in zip(negs['user_id'], negs['item_id']):
                    labels.append(user_item_sim[user][unlabeled_item])
            negs['label'] = labels
        negs = negs.merge(user_info[["user_id", INTERNAL_USER_ID_FIELD]], "left", on="user_id")
        negs = negs.merge(item_info[["item_id", INTERNAL_ITEM_ID_FIELD]], "left", on="item_id")
        negs = negs.drop(columns=["user_id", "item_id"])
        if "ref_item" in negs.columns:
            negs = negs.drop(columns=["ref_item"])        
        split_datasets['test'] = pd.concat([split_datasets['test'], negs])
        split_datasets['test'] = split_datasets['test'].sort_values(INTERNAL_USER_ID_FIELD).reset_index().drop(columns=['index'])

    for split in split_datasets.keys():
        split_datasets[split] = Dataset.from_pandas(split_datasets[split], preserve_index=False)

    return DatasetDict(split_datasets), Dataset.from_pandas(user_info, preserve_index=False), \
           Dataset.from_pandas(item_info, preserve_index=False), filtered_out_user_item_pairs_by_limit
