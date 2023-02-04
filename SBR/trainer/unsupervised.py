import os
import time
from os.path import exists, join

import torch
from tqdm import tqdm

from SBR.utils.metrics import calculate_metrics, log_results
from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD


class UnSupervisedTrainer:
    def __init__(self, config, model, device, logger, exp_dir,
                     relevance_level=1, users=None, items=None, dataset_eval_neg_sampling=None):
        self.model = model
        self.device = device
        self.logger = logger
        self.relevance_level = relevance_level
        neg_name = dataset_eval_neg_sampling['validation']
        if neg_name.startswith("f:"):
            neg_name = neg_name[len("f:"):]
        self.best_valid_output_path = {"ground_truth": join(exp_dir, f'best_valid_ground_truth_{neg_name}.json'),
                                       "predicted": join(exp_dir, f'best_valid_predicted_{neg_name}.json'),}
#                                       "log": join(exp_dir, f'best_valid_{neg_name}_log.txt')}
        neg_name = dataset_eval_neg_sampling['test']
        if neg_name.startswith("f:"):
            neg_name = neg_name[len("f:"):]
        self.test_output_path = {"ground_truth": join(exp_dir, f'test_ground_truth_{neg_name}.json'),
                                 "predicted": join(exp_dir, f'test_predicted_{neg_name}.json'),
                                 "log": join(exp_dir, f'test_{neg_name}_log.txt')}

        self.train_output_log = join(exp_dir, "outputs")
        os.makedirs(self.train_output_log, exist_ok=True)

        self.users = users
        self.items = items

        self.sig_output = config["sigmoid_output"]

        self.model.to(device)

    def evaluate(self, test_dataloader, valid_dataloader):
        outputs, ground_truth, internal_user_ids, internal_item_ids = self.predict(valid_dataloader)
        log_results(self.best_valid_output_path, ground_truth, outputs, internal_user_ids, internal_item_ids,
                    self.users, self.items)
        results = calculate_metrics(ground_truth, outputs, internal_user_ids, internal_item_ids, self.relevance_level, 0.5)
        results = {f"validation_{k}": v for k, v in results.items()}
        for k, v in results.items():
            self.logger.add_scalar(f'final_results/{k}', v)
        print(f"\nValidation results: {results}")

        outputs, ground_truth, internal_user_ids, internal_item_ids = self.predict(test_dataloader)
        log_results(self.test_output_path, ground_truth, outputs, internal_user_ids, internal_item_ids,
                    self.users, self.items)
        results = calculate_metrics(ground_truth, outputs, internal_user_ids, internal_item_ids, self.relevance_level, 0.5)
        results = {f"test_{k}": v for k, v in results.items()}
        for k, v in results.items():
            self.logger.add_scalar(f'final_results/{k}', v)
        print(f"\nTest results: {results}")

    def predict(self, eval_dataloader):
        # bring models to evaluation mode
        self.model.eval()

        outputs = []
        ground_truth = []
        user_ids = []
        item_ids = []
        pbar = tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), disable=False)

        start_time = time.perf_counter()
        with torch.no_grad():
            for batch_idx, batch in pbar:
                # data preparation
                batch = {k: v.to(self.device) for k, v in batch.items()}
                label = batch.pop("label").float()  # setting the type to torch.float32
                prepare_time = time.perf_counter() - start_time

                output = self.model(batch)
                if self.sig_output:
                    output = torch.sigmoid(output)

                process_time = time.perf_counter() - start_time - prepare_time
                proc_compute_efficiency = process_time / (process_time + prepare_time)

                ## for debugging. it needs access to actual user_id and item_id
                ground_truth.extend(label.squeeze().tolist())
                outputs.extend(output.squeeze().tolist())
                user_ids.extend(batch[
                                    INTERNAL_USER_ID_FIELD].squeeze().tolist())
                item_ids.extend(batch[INTERNAL_ITEM_ID_FIELD].squeeze().tolist())

                postprocess_time = time.perf_counter() - start_time - prepare_time - process_time
                pbar.set_description(
                    f'Compute efficiency: {proc_compute_efficiency:.4f}, '
                    f'prep: {prepare_time:.4f},'
                    f'process: {process_time:.4f}, post: {postprocess_time:.4f}')
                start_time = time.perf_counter()

        ground_truth = torch.tensor(ground_truth)
        outputs = torch.tensor(outputs)
        return outputs, ground_truth, user_ids, item_ids
