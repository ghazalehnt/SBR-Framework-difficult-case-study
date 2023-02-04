import argparse
import json
import os
import random
from os.path import join

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from SBR.trainer.supervised import SupervisedTrainer
from SBR.utils.data_loading import load_data
from SBR.utils.others import get_model
from SBR.trainer.unsupervised import UnSupervisedTrainer
from SBR.utils.statics import get_profile, map_user_item_text


def main(op, config_file=None, result_folder=None, given_user_text_filter=None, given_limit_training_data=None,
         given_neg_files=None, given_lr=None, given_tbs=None, given_user_text=None, given_item_text=None):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_only = False
    if op in ["train", "trainonly"]:
        config = json.load(open(config_file, 'r'))
        if given_user_text_filter is not None:
            config['dataset']['user_text_filter'] = given_user_text_filter
        if given_limit_training_data is not None:
            config['dataset']['limit_training_data'] = given_limit_training_data
        if given_lr is not None:
            config['trainer']['lr'] = given_lr
        if given_tbs is not None:
            config['dataset']['train_batch_size'] = given_tbs
        if given_user_text is not None:
            config['dataset']['user_text'] = get_profile(config['dataset']['name'], given_user_text)
        if given_item_text is not None:
            config['dataset']['item_text'] = get_profile(config['dataset']['name'], given_item_text)
        if "<DATA_ROOT_PATH" in config["dataset"]["dataset_path"]:
            DATA_ROOT_PATH = config["dataset"]["dataset_path"][config["dataset"]["dataset_path"].index("<"):
                             config["dataset"]["dataset_path"].index(">")+1]
            config["dataset"]["dataset_path"] = config["dataset"]["dataset_path"]\
                .replace(DATA_ROOT_PATH, open(f"data/paths_vars/{DATA_ROOT_PATH[1:-1]}").read().strip())
        if "<EXP_ROOT_PATH>" in config["experiment_root"]:
            config["experiment_root"] = config["experiment_root"]\
                .replace("<EXP_ROOT_PATH>", open("data/paths_vars/EXP_ROOT_PATH").read().strip())
        print(config)
        exp_dir_params = []
        for param in config['params_in_exp_dir']:
            p1 = param[:param.index(".")]
            p2 = param[param.index(".")+1:]
            if param == "dataset.validation_neg_sampling_strategy" and config[p1][p2].startswith("f:"):
                temp = config[p1][p2]
                temp = temp[temp.index("f:validation_neg_")+len("f:validation_neg_"):]
                exp_dir_params.append(f"f-{temp}")
            elif param == "dataset.test_neg_sampling_strategy" and config[p1][p2].startswith("f:"):
                temp = config[p1][p2]
                temp = temp[temp.index("f:test_neg_")+len("f:test_neg_"):]
                exp_dir_params.append(f"f-{temp}")
            elif isinstance(config[p1][p2], list):
                if p2 in ["item_text", "user_text"]:
                    exp_dir_params.append('-'.join([map_user_item_text[v] for v in config[p1][p2]]))
                else:
                    exp_dir_params.append('-'.join(config[p1][p2]))
            else:
                exp_dir_params.append(str(config[p1][p2]))
        exp_dir = join(config['experiment_root'], "_".join(exp_dir_params))
        
        config["experiment_dir"] = exp_dir
        # check if the exp dir exists, the config file is the same as given.
        if os.path.exists(join(exp_dir, "config.json")):
            config2 = json.load(open(join(exp_dir, "config.json"), 'r'))
#            # TODO: remove later, now for running experiments, enough logging:
#            config2["dataset"]["load_user_item_text"] =  config["dataset"]["load_user_item_text"] 
            if config != config2:
                raise ValueError(f"{exp_dir} exists with different config != {config_file}")
        os.makedirs(exp_dir, exist_ok=True)
        json.dump(config, open(join(exp_dir, "config.json"), 'w'), indent=4)
    elif op == "test":
        config = json.load(open(join(result_folder, "config.json"), 'r'))
        # TODO: this should also be removed, it was written to accomodate earlier runs
        if "chunk_size" in config["dataset"]:
            if "user_chunk_size" not in config["dataset"]:
                config["dataset"]["user_chunk_size"] = config["dataset"]["chunk_size"] 
            if "item_chunk_size" not in config["dataset"]:
                config["dataset"]["item_chunk_size"] = config["dataset"]["chunk_size"]
        ###
        test_only = True
        exp_dir = config["experiment_dir"]
        if given_neg_files["validation"] is not None:
            config["dataset"]["validation_neg_sampling_strategy"] = given_neg_files["validation"]
        if given_neg_files["test"] is not None:
            config["dataset"]["test_neg_sampling_strategy"] = given_neg_files["test"]
        config["dataset"]["load_user_item_text"] = False
    else:
        raise ValueError("op not defined!")

    # todo do we need all of these??? dts
    logger = SummaryWriter(exp_dir)
    for k, v in config["dataset"].items():
        logger.add_text(f"dataset/{k}", str(v))
    for k, v in config["trainer"].items():
        logger.add_text(f"trainer/{k}", str(v))
    for k, v in config["model"].items():
        logger.add_text(f"model/{k}", str(v))
    logger.add_text("exp_dir", exp_dir)
    print("experiment_dir:")
    print(exp_dir)

#    # TODO: remove later, now for running experiments, enough logging:
#    config["dataset"]["load_user_item_text"] = False

    train_dataloader, valid_dataloader, test_dataloader, users, items, relevance_level, padding_token = \
        load_data(config['dataset'],
                  config['model']['pretrained_model'] if 'pretrained_model' in config['model'] else None)
    print("Data load done!")
    # needed for item-item relatedness
    temp = {ex: internal for ex, internal in zip(items['item_id'], items['internal_item_id'])}
    json.dump(temp, open(join(exp_dir, "item_internal_ids.json"), 'w'))

    model = get_model(config['model'], users, items, device, config['dataset'])
    print("Get model Done!")

    if config['trainer']['optimizer'] == "":
        trainer = UnSupervisedTrainer(config=config['trainer'], model=model, device=device, logger=logger,
                                    exp_dir=exp_dir,
                                    relevance_level=relevance_level,
                                    users=users, items=items,
                                    dataset_eval_neg_sampling=
                                    {"validation": config["dataset"]["validation_neg_sampling_strategy"],
                                     "test": config["dataset"]["test_neg_sampling_strategy"]})
        trainer.evaluate(test_dataloader, valid_dataloader)
    else:
        trainer = SupervisedTrainer(config=config['trainer'], model=model, device=device, logger=logger, exp_dir=exp_dir,
                                test_only=test_only, relevance_level=relevance_level,
                                users=users, items=items,
                                dataset_eval_neg_sampling=
                                {"validation": config["dataset"]["validation_neg_sampling_strategy"],
                                 "test": config["dataset"]["test_neg_sampling_strategy"]})
        if op == "train":
            trainer.fit(train_dataloader, valid_dataloader)
            trainer.evaluate(test_dataloader, valid_dataloader)
        elif op == "trainonly":
            trainer.fit(train_dataloader, valid_dataloader)
        elif op == "test":
            trainer.evaluate(test_dataloader, valid_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', '-c', type=str, default=None, help='config file, to train')
    parser.add_argument('--result_folder', '-r', type=str, default=None, help='result forler, to evaluate')
    parser.add_argument('--user_text_filter', type=str, default=None, help='user_text_filter used only if given, otherwise read from the config')
    parser.add_argument('--limit_training_data', '-l', type=str, default=None, help='the file name containing the limited training data')
    parser.add_argument('--testtime_validation_neg_strategy', '-v', default=None, help='valid neg strategy, only for op == test')
    parser.add_argument('--testtime_test_neg_strategy', '-t', default=None, help='test neg strategy, only for op == test')
    parser.add_argument('--trainer_lr', default=None, help='trainer learning rate')
    parser.add_argument('--train_batch_size', default=None, help='train_batch_size')
    parser.add_argument('--user_text', default=None, help='user_text (tg,tgr,tc,tcsr)')
    parser.add_argument('--item_text', default=None, help='item_text (tg,tgd,tc,tcd)')
    parser.add_argument('--op', type=str, help='operation train/test/trainonly')
    args, _ = parser.parse_known_args()

    if args.op in ["train", "trainonly"]:
        if not os.path.exists(args.config_file):
            raise ValueError(f"Config file does not exist: {args.config_file}")
        if args.result_folder:
            raise ValueError(f"OP==train does not accept result_folder")
        if args.testtime_validation_neg_strategy or args.testtime_test_neg_strategy:
            raise ValueError(f"OP==train does not accept test-time eval neg strategies.")
        main(op=args.op, config_file=args.config_file, given_user_text_filter=args.user_text_filter,
             given_limit_training_data=args.limit_training_data,
             given_lr=float(args.trainer_lr) if args.trainer_lr is not None else args.trainer_lr,
             given_tbs=int(args.train_batch_size) if args.train_batch_size is not None else args.train_batch_size,
             given_user_text=args.user_text, given_item_text=args.item_text)
    elif args.op == "test":
        if not os.path.exists(join(args.result_folder, "config.json")):
            raise ValueError(f"Result folder does not exist: {args.config_file}")
        if args.config_file:
            raise ValueError(f"OP==test does not accept config_file")
        main(op=args.op, result_folder=args.result_folder,
             given_neg_files={"validation": args.testtime_validation_neg_strategy,
                              "test": args.testtime_test_neg_strategy})


