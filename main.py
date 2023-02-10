import argparse
import json
import os
import random
from os.path import join

import torch
import numpy as np

from SBR.trainer.supervised import SupervisedTrainer
from SBR.utils.data_loading import load_data
from SBR.utils.others import get_model
from SBR.utils.statics import get_profile, map_user_item_text


def main(op, config_file=None, result_folder=None, given_test_neg_file=None,
         given_lr=None, given_tbs=None, given_user_text=None, given_item_text=None):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_only = False
    if op in ["train"]:
        config = json.load(open(config_file, 'r'))
        if given_lr is not None:
            config['trainer']['lr'] = given_lr
        if given_tbs is not None:
            config['dataset']['train_batch_size'] = given_tbs
        if given_user_text is not None:
            config['dataset']['user_text'] = get_profile(config['dataset']['name'], given_user_text)
        if given_item_text is not None:
            config['dataset']['item_text'] = get_profile(config['dataset']['name'], given_item_text)
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
            if config != config2:
                raise ValueError(f"{exp_dir} exists with different config != {config_file}")
        os.makedirs(exp_dir, exist_ok=True)
        json.dump(config, open(join(exp_dir, "config.json"), 'w'), indent=4)
    elif op == "test":
        config = json.load(open(join(result_folder, "config.json"), 'r'))
        test_only = True
        exp_dir = config["experiment_dir"]
        if given_test_neg_file is not None:
            config["dataset"]["test_neg_sampling_strategy"] = given_test_neg_file
        config["dataset"]["load_user_item_text"] = False
    else:
        raise ValueError("op not defined!")

    print("experiment_dir:")
    print(exp_dir)

    train_dataloader, valid_dataloader, test_dataloader, users, items, relevance_level, padding_token = \
        load_data(config['dataset'],
                  config['model']['pretrained_model'] if 'pretrained_model' in config['model'] else None)
    print("Data load done!")
    # needed for item-item relatedness
    temp = {ex: internal for ex, internal in zip(items['item_id'], items['internal_item_id'])}
    json.dump(temp, open(join(exp_dir, "item_internal_ids.json"), 'w'))

    model = get_model(config['model'], users, items, device, config['dataset'])
    print("Get model Done!")

    trainer = SupervisedTrainer(config=config['trainer'], model=model, device=device, exp_dir=exp_dir,
                            test_only=test_only, relevance_level=relevance_level,
                            users=users, items=items,
                            test_eval_file_name=config["dataset"]["test_neg_sampling_strategy"] if "test_neg_sampling_strategy" in config["dataset"] else None)
    if op == "train":
        trainer.fit(train_dataloader, valid_dataloader)
    elif op == "test":
        trainer.evaluate(test_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', '-c', type=str, default=None, help='config file, to train')
    parser.add_argument('--result_folder', '-r', type=str, default=None, help='result forler, to evaluate')
    parser.add_argument('--testtime_test_neg_strategy', '-t', default=None, help='test neg strategy, only for op == test')
    parser.add_argument('--trainer_lr', default=None, help='trainer learning rate')
    parser.add_argument('--train_batch_size', default=None, help='train_batch_size')
    parser.add_argument('--user_text', default=None, help='user_text (tg,tgr,tc,tcsr)')
    parser.add_argument('--item_text', default=None, help='item_text (tg,tgd,tc,tcd)')
    parser.add_argument('--op', type=str, help='operation train/test/trainonly')
    args, _ = parser.parse_known_args()

    if args.op in ["train"]:
        if not os.path.exists(args.config_file):
            raise ValueError(f"Config file does not exist: {args.config_file}")
        if args.result_folder:
            raise ValueError(f"OP==train does not accept result_folder")
        if args.testtime_validation_neg_strategy or args.testtime_test_neg_strategy:
            raise ValueError(f"OP==train does not accept test-time eval neg strategies.")
        main(op=args.op, config_file=args.config_file,
             given_lr=float(args.trainer_lr) if args.trainer_lr is not None else args.trainer_lr,
             given_tbs=int(args.train_batch_size) if args.train_batch_size is not None else args.train_batch_size,
             given_user_text=args.user_text, given_item_text=args.item_text)
    elif args.op == "test":
        if not os.path.exists(join(args.result_folder, "config.json")):
            raise ValueError(f"Result folder does not exist: {args.config_file}")
        if args.config_file:
            raise ValueError(f"OP==test does not accept config_file")
        main(op=args.op, result_folder=args.result_folder,
             given_test_neg_file=args.testtime_test_neg_strategy)


