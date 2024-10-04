from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
import random
import sys
import shutil
from transformers import set_seed
import argparse

import os, copy, tqdm
from torch import nn
import torch.nn.functional as F

accum = {}

class FisherTrainer(Trainer):
    def create_optimizer(self):
        self.optimizer = torch.optim.SGD(self.model_wrapped.parameters(), lr=0.0)
        return self.optimizer
    
    def training_step(self, model, inputs):
        output = super().training_step(model, inputs)
        with torch.no_grad():
            for k, v in model.named_parameters():
                if (('bert.encoder.layer' in k and 'weight' in k) and len(v.shape) >= 2):
                    accum[k] = accum.get(k, 0.) + (v.grad.detach().double() ** 2)
        return output
        
def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='cola', type=str, help="target subtask in the GLUE benchmark")
    parser.add_argument("--lr", default=2e-5, type=float, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=0., help="regularization coefficient")
    parser.add_argument("--num-epochs", type=int, default=2, help="number of training epochs")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id for training")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    GLUE_TASKS = ["cola", "mnli", "mrpc", "qnli", "qqp", "sst2", "stsb"]
    task = args.task
    save_path = "./"
    model_checkpoint = "bert-base-uncased"
    lr = args.lr
    seed = args.seed
    wd = args.weight_decay
    num_epochs = args.num_epochs
    batch_size = 32
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    set_seed(seed)

    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("glue", actual_task)
    metric = load_metric('glue', actual_task)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
    sentence1_key, sentence2_key = task_to_keys[task]
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
    model_name = model_checkpoint.split("/")[-1]

    args = TrainingArguments(
        f"{save_path}/{task}-{lr}-{wd}-{batch_size}-{num_epochs}-{seed}",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        logging_strategy = "epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=wd,
        num_train_epochs=num_epochs,
        seed=seed,
        full_determinism=True
    )

    validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    original_state_dict = copy.deepcopy(model.state_dict())
    
    args_fisher = TrainingArguments(
        f"{save_path}/fisher",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        logging_strategy = "epoch",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=lr,
        lr_scheduler_type='constant',
        weight_decay=0.00,
        num_train_epochs=1,
        seed=seed,
        full_determinism=True
    )
    
    trainer_fisher = FisherTrainer(
        model,
        args_fisher,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    trainer_fisher.train()

    os.makedirs(f'{save_path}/fisher_{task}-{lr}-{wd}-{batch_size}-{num_epochs}-{seed}', exist_ok=True)
    for k, v in tqdm.tqdm(original_state_dict.items()):
        if k in accum:
            torch.save(accum[k], f'{save_path}/fisher_{task}-{lr}-{wd}-{batch_size}-{num_epochs}-{seed}/{k}.pkt')
            
    shutil.rmtree(f"{save_path}/fisher")