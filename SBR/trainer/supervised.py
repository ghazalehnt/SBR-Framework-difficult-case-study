import operator
import os
import time
from os.path import exists, join

import torch
from ray import tune
from torch.optim import Adam, SGD
from tqdm import tqdm
import numpy as np

from SBR.utils.metrics import calculate_metrics, log_results
from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD


class SupervisedTrainer:
    def __init__(self, config, model, device, exp_dir, test_only=False, tuning=False, save_checkpoint=True,
                 relevance_level=1, users=None, items=None, test_eval_file_name=None):
        self.model = model
        self.device = device
        self.test_only = test_only
        self.tuning = tuning
        self.save_checkpoint = save_checkpoint
        self.relevance_level = relevance_level
        self.valid_metric = config['valid_metric']
        self.patience = config['early_stopping_patience']
        self.best_model_path = join(exp_dir, 'best_model.pth')
        if test_eval_file_name is not None:
            neg_name = test_eval_file_name
            if neg_name.startswith("f:"):
                neg_name = neg_name[len("f:"):]
            self.test_output_path = {"ground_truth": join(exp_dir, f'test_ground_truth_{neg_name}.json'),
                                     "predicted": join(exp_dir, f'test_predicted_{neg_name}'),}
    #                                 "log": join(exp_dir, f'test_{neg_name}_log_100users')}

        self.users = users
        self.items = items
        self.sig_output = config["sigmoid_output"]

        if config['loss_fn'] == "BCE":
            if self.sig_output is False:
                raise ValueError("cannot have BCE with no sigmoid")
            self.loss_fn = torch.nn.BCEWithLogitsLoss()  # use BCEWithLogitsLoss and do not apply the sigmoid beforehand
        elif config['loss_fn'] == "MRL":
            self.loss_fn = torch.nn.MarginRankingLoss()
        elif config['loss_fn'] == "MSE":
            self.loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError(f"loss_fn {config['loss_fn']} is not implemented!")

        self.epochs = config['epochs']
        self.start_epoch = 0
        self.best_epoch = 0
        self.best_saved_valid_metric = np.inf if self.valid_metric == "valid_loss" else -np.inf
        if exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_epoch = checkpoint['epoch']
            self.best_saved_valid_metric = checkpoint['best_valid_metric']
            print("last checkpoint restored")
        self.model.to(device)
        
        if not test_only:
            if config['optimizer'] == "Adam":
                self.optimizer = Adam(self.model.parameters(), lr=config['lr'], weight_decay=config['wd'])
            elif config['optimizer'] == "SGD":
                self.optimizer = SGD(self.model.parameters(), lr=config['lr'], weight_decay=config['wd'], momentum=config['momentum'], nesterov=config['nesterov'])
            else:
                raise ValueError(f"Optimizer {config['optimizer']} not implemented!")
            if exists(self.best_model_path):
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def fit(self, train_dataloader, valid_dataloader):
        early_stopping_cnt = 0
        comparison_op = operator.lt if self.valid_metric == "valid_loss" else operator.gt

        for epoch in range(self.start_epoch, self.epochs):
            if early_stopping_cnt == self.patience:
                print(f"Early stopping after {self.patience} epochs not improving!")
                break

            self.model.train()
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=True if self.tuning else False)
            start_time = time.perf_counter()
            train_loss, total_count = 0, 0

            for batch_idx, batch in pbar:
                # data preparation
                batch = {k: v.to(self.device) for k, v in batch.items()}
                label = batch.pop("label").float()  # setting the type to torch.float32
                prepare_time = time.perf_counter() - start_time

                self.optimizer.zero_grad()
                output = self.model(batch)
                if self.loss_fn._get_name() == "BCEWithLogitsLoss":
                    # not applying sigmoid before loss bc it is already applied in the loss
                    loss = self.loss_fn(output, label)
                else:
                    if self.sig_output:
                        output = torch.sigmoid(output)
                    if self.loss_fn._get_name() == "MarginRankingLoss":
                        pos_l = label[label == 1]
                        pos_out = output[:pos_l.shape[0]].squeeze()
                        neg_out = output[pos_l.shape[0]:].squeeze()
                        loss = self.loss_fn(pos_out, neg_out, pos_l)
                    else:
                        loss = self.loss_fn(output, label)

                loss.backward()
                self.optimizer.step()
                train_loss += loss
                total_count += label.size(0)

                # compute computation time and *compute_efficiency*
                process_time = time.perf_counter() - start_time - prepare_time
                compute_efficiency = process_time / (process_time + prepare_time)
                pbar.set_description(
                    f'Compute efficiency: {compute_efficiency:.4f}, '
                    f'loss: {loss.item():.8f},  epoch: {epoch}/{self.epochs}'
                    f'prep: {prepare_time:.4f}, process: {process_time:.4f}')
                start_time = time.perf_counter()
            train_loss /= total_count
            print(f"Train loss epoch {epoch}: {train_loss}")

            outputs, ground_truth, valid_loss, users, items = self.predict(valid_dataloader, low_mem=True)
            results = calculate_metrics(ground_truth, outputs, users, items, self.relevance_level)
            results["loss"] = valid_loss
            results = {f"valid_{k}": v for k, v in results.items()}
            print(f"Valid loss epoch {epoch}: {valid_loss} - {self.valid_metric} = {results[self.valid_metric]:.6f}\n")

            if comparison_op(results[self.valid_metric], self.best_saved_valid_metric):
                self.best_saved_valid_metric = results[self.valid_metric]
                self.best_epoch = epoch
                if self.save_checkpoint:
                    checkpoint = {
                        'epoch': self.best_epoch,
                        'best_valid_metric': self.best_saved_valid_metric,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        }
                    torch.save(checkpoint, f"{self.best_model_path}_tmp")
                    os.rename(f"{self.best_model_path}_tmp", self.best_model_path)
                early_stopping_cnt = 0
            else:
                early_stopping_cnt += 1
            # report to tune
            if self.tuning:
                tune.report(best_valid_metric=self.best_saved_valid_metric,
                            best_epoch=self.best_epoch,
                            epoch=epoch)

    def evaluate(self, test_dataloader):
        # load the best model from file.
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.best_epoch = checkpoint['epoch']
        self.best_saved_valid_metric = checkpoint['best_valid_metric']
        print("best model loaded!")

        outputs, ground_truth, test_loss, internal_user_ids, internal_item_ids = self.predict(test_dataloader)
        log_results(ground_truth, outputs, internal_user_ids, internal_item_ids,
                    self.users, self.items,
                    self.test_output_path['ground_truth'],
                    f"{self.test_output_path['predicted']}_e-{self.best_epoch}.json",
                    f"{self.test_output_path['log']}_e-{self.best_epoch}.txt" if "log" in self.test_output_path else None)

    def predict(self, eval_dataloader, low_mem=False):
        # bring models to evaluation mode
        self.model.eval()

        outputs = []
        ground_truth = []
        user_ids = []
        item_ids = []
        eval_loss, total_count = 0, 0
        pbar = tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), disable=True if self.tuning else False)

        start_time = time.perf_counter()
        with torch.no_grad():
            for batch_idx, batch in pbar:
                # data preparation
                batch = {k: v.to(self.device) for k, v in batch.items()}
                label = batch.pop("label").float()  # setting the type to torch.float32
                prepare_time = time.perf_counter() - start_time

                output = self.model(batch)
                if self.loss_fn._get_name() == "BCEWithLogitsLoss":
                    # not applying sigmoid before loss bc it is already applied in the loss
                    loss = self.loss_fn(output, label)
                    # just apply sigmoid for logging
                    output = torch.sigmoid(output)
                else:
                    if self.sig_output:
                        output = torch.sigmoid(output)
                    if self.loss_fn._get_name() == "MarginRankingLoss":
                        loss = torch.Tensor([-1])  # cannot calculate margin loss with more than 1 negative per positve
                    else:
                        loss = self.loss_fn(output, label)
                eval_loss += loss.item()
                total_count += label.size(0)  # TODO remove if not used
                process_time = time.perf_counter() - start_time - prepare_time
                proc_compute_efficiency = process_time / (process_time + prepare_time)

                ## for debugging. it needs access to actual user_id and item_id
                ground_truth.extend(label.squeeze().tolist())
                outputs.extend(output.squeeze().tolist())
                user_ids.extend(batch[
                                    INTERNAL_USER_ID_FIELD].squeeze().tolist())
                if not low_mem:
                    item_ids.extend(batch[INTERNAL_ITEM_ID_FIELD].squeeze().tolist())

                postprocess_time = time.perf_counter() - start_time - prepare_time - process_time
                pbar.set_description(
                    f'Compute efficiency: {proc_compute_efficiency:.4f}, '
                    f'loss: {loss.item():.8f},  prep: {prepare_time:.4f},'
                    f'process: {process_time:.4f}, post: {postprocess_time:.4f}')
                start_time = time.perf_counter()

            eval_loss /= total_count
        ground_truth = torch.tensor(ground_truth)
        outputs = torch.tensor(outputs)
        return outputs, ground_truth, eval_loss, user_ids, item_ids
