import argparse
import json
import os
import random
from os.path import exists, join

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import numpy as np
import torch
from tensorboardX import SummaryWriter

from SBR.utils.data_loading import load_data
from SBR.utils.others import get_model
from SBR.trainer.supervised import SupervisedTrainer
from SBR.utils.statics import get_profile, map_user_item_text


def training_function(tuning_config, stationary_config_file, exp_root_dir, data_root_dir,
                      valid_metric, early_stopping_patience=None, save_checkpoint=False,
                      given_user_text=None, given_item_text=None, given_user_text_filter=None):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")

    config = json.load(open(stationary_config_file, 'r'))
    for k in tuning_config:
        if '.' in k:
            l1 = k[:k.index(".")]
            l2 = k[k.index(".")+1:]
            config[l1][l2] = tuning_config[k]
            if l2 == "max_num_chunks":
                config[l1]['max_num_chunks_user'] = tuning_config[k]
                config[l1]['max_num_chunks_item'] = tuning_config[k]
        else:
            config[k] = tuning_config[k]
    if "<DATA_ROOT_PATH" in config["dataset"]["dataset_path"]:
        DATA_ROOT_PATH = config["dataset"]["dataset_path"][config["dataset"]["dataset_path"].index("<"):
                                                           config["dataset"]["dataset_path"].index(">") + 1]
        config["dataset"]["dataset_path"] = config["dataset"]["dataset_path"] \
            .replace(DATA_ROOT_PATH, data_root_dir)
    if "<EXP_ROOT_PATH>" in config["experiment_root"]:
        config["experiment_root"] = config["experiment_root"] \
            .replace("<EXP_ROOT_PATH>", exp_root_dir)

    if given_user_text is not None:
        config['dataset']['user_text'] = get_profile(config['dataset']['name'], given_user_text)
    if given_item_text is not None:
        config['dataset']['item_text'] = get_profile(config['dataset']['name'], given_item_text)
    if given_user_text_filter is not None:
        config['dataset']['user_text_filter'] = given_user_text_filter

    exp_dir_params = []
    for param in config['params_in_exp_dir']:
        p1 = param[:param.index(".")]
        p2 = param[param.index(".") + 1:]
        if param == "dataset.validation_neg_sampling_strategy" and config[p1][p2].startswith("f:"):
            temp = config[p1][p2]
            temp = temp[temp.index("f:validation_neg_") + len("f:validation_neg_"):]
            exp_dir_params.append(f"f-{temp}")
        elif param == "dataset.test_neg_sampling_strategy" and config[p1][p2].startswith("f:"):
            temp = config[p1][p2]
            temp = temp[temp.index("f:test_neg_") + len("f:test_neg_"):]
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
    if early_stopping_patience is not None:
        config['trainer']['early_stopping_patience'] = early_stopping_patience
    config['trainer']['valid_metric'] = valid_metric
    print(config)

    if exists(exp_dir):
        saved_config = json.load(open(os.path.join(exp_dir, "config.json"), 'r'))
        if saved_config != config:
            raise ValueError(f"Given config should be the same as saved config file!!!!{exp_dir}")
    else:
        os.makedirs(exp_dir)
        with open(os.path.join(exp_dir, "config.json"), 'w') as log:
            json.dump(config, log, indent=4)

    # log into the ray_tune trial folder
    with open("config.json", 'w') as log:
        json.dump(config, log, indent=4)

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

    train_dataloader, valid_dataloader, test_dataloader, users, items, relevance_level, padding_token = \
        load_data(config['dataset'],
                  config['model']['pretrained_model'] if 'pretrained_model' in config['model'] else None)

    model = get_model(config['model'], users, items, device, config['dataset'])

    trainer = SupervisedTrainer(config=config['trainer'], model=model, device=device, logger=logger, exp_dir=exp_dir,
                                test_only=False, tuning=True,
                                relevance_level=relevance_level,
                                users=users, items=items,
                                dataset_eval_neg_sampling=
                                {"validation": config["dataset"]["validation_neg_sampling_strategy"],
                                 "test": config["dataset"]["test_neg_sampling_strategy"]},
                                save_checkpoint=save_checkpoint)
    trainer.fit(train_dataloader, valid_dataloader)


def main(hyperparameter_config, config_file, ray_result_dir, name, valid_metric, max_epochs=50, grace_period=5, num_gpus_per_trial=0,
         num_cpus_per_trial=2, extra_gpus=0, num_samples=1, resume=False, save_checkpoint=False,
         early_stopping_patience=None, num_concurrent=1, data_name="GR",
         given_user_text=None, given_item_text=None, given_user_text_filter=None):
    exp_root_dir = open("data/paths_vars/EXP_ROOT_PATH").read().strip()
    data_root_dir = open(f"data/paths_vars/DATA_ROOT_PATH_{data_name}").read().strip()
    if "<EXP_ROOT_PATH>" in ray_result_dir:
        ray_result_dir = ray_result_dir.replace("<EXP_ROOT_PATH>", exp_root_dir)
    print(f"ray dir: {ray_result_dir}")
    scheduler = ASHAScheduler(
        metric="best_valid_metric",
        mode="min" if "loss" in valid_metric else "max",
        max_t=max_epochs,
        grace_period=grace_period
    )
    reporter = CLIReporter(
        metric_columns=["epoch", "best_valid_metric", "best_epoch"], max_report_frequency=3600
    )
    result = tune.run(
        tune.with_parameters(training_function, stationary_config_file=config_file,
                             valid_metric=valid_metric, early_stopping_patience=early_stopping_patience,
                             exp_root_dir=exp_root_dir, data_root_dir=data_root_dir,
                             save_checkpoint=save_checkpoint,
                             given_item_text=given_item_text, given_user_text=given_user_text,
                             given_user_text_filter=given_user_text_filter),
        name=name,
        resources_per_trial={"cpu": num_cpus_per_trial, "gpu": num_gpus_per_trial, "extra_gpu": extra_gpus},
        config=hyperparameter_config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=ray_result_dir,
        resume=resume,
        max_concurrent_trials=num_concurrent  # todo remove - set 1 for debug
    )
    best_trial = result.get_best_trial("best_valid_metric", "min" if "loss" in valid_metric else "max", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final best_valid_metric({valid_metric}): {best_trial.last_result['best_valid_metric']} - "
          f"best_epoch: {best_trial.last_result['best_epoch']}, last_epoch: {best_trial.last_result['epoch']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the parameter tuning pipeline.')
    parser.add_argument('--config_file', type=str, help='path to configuration file.')
    parser.add_argument('--hyperparam_config_file', type=str, help='path to hyperparam configuration file.')
    parser.add_argument('--num_gpu', type=int, help='number of gpus.')
#    parser.add_argument('--num_cpu', type=int, help='number of cpus.')
    parser.add_argument('--num_con', type=int, help='number of concurrent.')
    parser.add_argument('--user_text', default=None, help='user_text (tg,tgr,tc,tcsr)')
    parser.add_argument('--item_text', default=None, help='item_text (tg,tgd,tc,tcd)')
    parser.add_argument('--user_text_filter', type=str, default=None, help='user_text_filter used only if given, otherwise read from the config')

    args = parser.parse_args()
    if not exists(args.config_file):
        raise ValueError(f"File: {args.config_file} does not exist!")
    if not exists(args.hyperparam_config_file):
        raise ValueError(f"File: {args.hyperparam_config_file} does not exist!")

    config = json.load(open(args.hyperparam_config_file, 'r'))

    hyper_config = {}
    for k, v in config["space"].items():
        if 'grid_search' in v:
            hyper_config[k] = tune.grid_search(v['grid_search'])
        elif 'quniform' in v:
            hyper_config[k] = tune.quniform(v['quniform'][0], v['quniform'][1], v['quniform'][2])
        elif 'loguniform' in v:
            hyper_config[k] = tune.loguniform(v['loguniform'][0], v['loguniform'][1])
        elif 'choice' in v:
            hyper_config[k] = tune.choice(v['choice'])
        else:
            raise NotImplemented("implement different space types")

    ray.init(num_gpus=args.num_gpu)  #, num_cpus=args.num_cpu)
    main(hyper_config, os.path.abspath(args.config_file), config['ray_result_dir'],
         config['name'], config['valid_metric'],
         max_epochs=config['max_epochs'], grace_period=config['grace_period'],
         num_gpus_per_trial=config['num_gpus_per_trial'], num_cpus_per_trial=config["num_cpus_per_trial"],
         num_samples=config['num_samples'], resume=config['resume'], save_checkpoint=config['save_checkpoint'],
         early_stopping_patience=config['early_stopping_patience'] if "early_stopping_patience" in config else None,
         num_concurrent=args.num_con, data_name=config['data_name'],
         given_user_text=args.user_text, given_item_text=args.item_text,
         given_user_text_filter=args.user_text_filter)
