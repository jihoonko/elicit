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
import pickle

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

class ELiCiTInner(nn.Module):
    def __init__(self, num_feats, dims, eps=1e-12):
        super().__init__()
        self.num_feats = num_feats
        num_dims = len(dims)
        self.num_dims = num_dims
        self.num_values = (1 << num_dims)
        # the corresponding values of the reference states
        self.values = nn.Parameter(torch.empty(1, self.num_values, num_feats))
        with torch.no_grad():
            scale = .01 / (self.num_feats ** 0.5)
            nn.init.trunc_normal_(self.values, 0, scale)
            
    def forward(self, probs):
        return probs

class ELiCiT(nn.Module):
    def __init__(self, dims, num_feats, qlevel=4):
        super().__init__()
        self.num_feats = num_feats
        self.dims = dims
        self.qlevel = qlevel
        self.order = len(self.dims)
        # for handling the corresponding values of the reference states
        self.inner = ELiCiTInner(num_feats=num_feats, dims=dims)
        # features of the indices
        self.feats = nn.Parameter(torch.rand(sum(self.dims), self.num_feats))
        # candidate values
        if qlevel != -1:
            self.candidates = nn.Parameter(torch.linspace(1. / (1 << (self.qlevel + 1)), 1. - 1. / (1 << (self.qlevel + 1)), steps=(1 << self.qlevel)).view(1, 1, (1 << self.qlevel)).repeat(self.order, self.num_feats, 1))
        else:
            self.candidates = nn.Parameter(torch.zeros(1))

        # scale and bias to stablize the training
        self.scale = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer('which_axis', torch.cat([i * torch.ones(dim, dtype=torch.long) for i, dim in enumerate(self.dims)], dim=-1))
        
    def prepare_inputs(self):
        device = self.feats.data.device
        if self.qlevel == -1:
            probs = self.feats
        else:
            target_keys = self.candidates[self.which_axis]
            target_feats = self.feats
            # find the target candidate
            assigned = torch.argmin((target_feats.unsqueeze(-1) - target_keys).abs(), dim=-1).unsqueeze(-1)
            # straight-through technique
            probs = -(target_feats.detach()) + (torch.gather(target_keys, -1, assigned).squeeze(-1) + target_feats)
        return torch.stack((probs, 1. - probs), dim=0)

    def forward(self):
        # fast computation with matrix multiplication operations (for 2-order tensors)
        mats = self.inner.values.view(4, self.num_feats // 2, 2).permute(2, 0, 1) # 2 x 4 x 1 x feat/2
        row_feats, col_feats = torch.split(self.prepare_inputs(), self.dims, dim=1) # 2 x dim x feat
        row_feats = row_feats.repeat_interleave(2, dim=0).view(4, self.dims[0], 2, self.num_feats // 2).permute(2, 0, 1, 3) # 2 x 4 x dim x feat/2
        col_feats = col_feats.view(2, self.dims[1], 2, self.num_feats // 2).repeat(2, 1, 1, 1).permute(2, 0, 3, 1) # 2 x 4 x feat/2 x dim
        ready = torch.matmul(mats.unsqueeze(-2) * row_feats, col_feats).sum(1) # 2 x dim1 x dim2
        preds = (ready[0] * F.tanh(ready[1])) * torch.exp(self.scale) + self.bias
        return preds

def get_matrices(name, target, rank, weights=None):
    if weights is None:
        weights = torch.ones_like(target)
        
    our_model = ELiCiT(target.shape, rank).to(0)
    optimizer = torch.optim.Adam([{'params': our_model.inner.parameters(), 'lr': 1e-3}, {'params': [our_model.feats, our_model.candidates, our_model.scale, our_model.bias], 'lr': 1e-2}], lr=1e-2)
    best_score = 1e10
    best_mat = copy.deepcopy(our_model.state_dict())
    patience = 0
    epoch = 0
    num_epochs = 5000
    while epoch <= num_epochs:
        optimizer.zero_grad()
        preds = our_model()
        (weights * (((target - preds) ** 2))).sum().backward()
        optimizer.step()
        
        preds = our_model()
        loss = (weights * (((target - preds) ** 2))).sum()
        
        epoch += 1
        if best_score >= loss.item():
            best_score = loss.item()
            best_mat = copy.deepcopy(our_model.state_dict())
            
    return best_mat

class NewLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features, out_features, rank, pretrained_bias, bias = True):
        super(NewLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.dims = (out_features, in_features)
        self.w = ELiCiT(self.dims, rank, qlevel=4)
        if bias:
            self.bias = nn.Parameter(pretrained_bias.data.detach())
        else:
            self.register_parameter('bias', None)
        
    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        self_weight = self.w()
        return F.linear(_input, self_weight, self.bias)
        
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
        
class DecompLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features, out_features, rank, initial_w1, initial_w2, initial_b, bias = True):
        super(DecompLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.dims = (out_features, in_features)
        self.w1 = nn.Parameter(initial_w1.detach())
        self.w2 = nn.Parameter(initial_w2.detach())
        if bias:
            self.bias = nn.Parameter(initial_b.detach())
        else:
            self.register_parameter('bias', None)
        
    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        self_weight = torch.matmul(self.w1, self.w2)
        return F.linear(_input, self_weight, self.bias)
        
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='cola', type=str, help="target subtask in the GLUE benchmark")
    parser.add_argument("--target", default='cola-2e-05-0.0-32-2-0/checkpoint-536', type=str, help="target checkpoint path")
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
        "mnli-mm": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
    metric_names = {
        "cola": 'matthews_correlation',
        "mnli": 'accuracy',
        "mnli-mm": 'accuracy',
        "mrpc": 'f1',
        "qnli": 'accuracy',
        "qqp": 'f1',
        "sst2": 'accuracy',
        "stsb": 'pearson',
    }
    
    finetuned_checkpoint = f'{save_path}/{args.target}'
    
    sentence1_key, sentence2_key = task_to_keys[task]
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
    model = AutoModelForSequenceClassification.from_pretrained(finetuned_checkpoint, num_labels=num_labels)

    metric_name = metric_names[task]
    model_name = model_checkpoint.split("/")[-1]

    ft_args = TrainingArguments(
        f"finetune_ours",
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
    original_state_dict = copy.deepcopy(model.state_dict())

    trainer = Trainer(
        model,
        ft_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    decomposed_state_dict = torch.load(f'{save_path}/{args.target.split('/')[0]}.pkt')
    rank = 256
    for i in range(12):
        model.bert.encoder.layer[i].attention.output.dense = NewLinear(768, 768, rank, original_state_dict[f'bert.encoder.layer.{i}.attention.output.dense.bias']).to(0)
        model.bert.encoder.layer[i].attention.self.key = NewLinear(768, 768, rank, original_state_dict[f'bert.encoder.layer.{i}.attention.self.key.bias']).to(0)
        model.bert.encoder.layer[i].attention.self.query = NewLinear(768, 768, rank, original_state_dict[f'bert.encoder.layer.{i}.attention.self.query.bias']).to(0)
        model.bert.encoder.layer[i].attention.self.value = NewLinear(768, 768, rank, original_state_dict[f'bert.encoder.layer.{i}.attention.self.value.bias']).to(0)
        model.bert.encoder.layer[i].intermediate.dense = NewLinear(768, 3072, rank, original_state_dict[f'bert.encoder.layer.{i}.intermediate.dense.bias']).to(0)
        model.bert.encoder.layer[i].output.dense = NewLinear(3072, 768, rank, original_state_dict[f'bert.encoder.layer.{i}.output.dense.bias']).to(0)

        model.bert.encoder.layer[i].attention.output.dense.w.load_state_dict(decomposed_state_dict[f'bert.encoder.layer.{i}.attention.output.dense.weight'])
        model.bert.encoder.layer[i].attention.self.key.w.load_state_dict(decomposed_state_dict[f'bert.encoder.layer.{i}.attention.self.key.weight'])
        model.bert.encoder.layer[i].attention.self.query.w.load_state_dict(decomposed_state_dict[f'bert.encoder.layer.{i}.attention.self.query.weight'])
        model.bert.encoder.layer[i].attention.self.value.w.load_state_dict(decomposed_state_dict[f'bert.encoder.layer.{i}.attention.self.value.weight'])
        model.bert.encoder.layer[i].intermediate.dense.w.load_state_dict(decomposed_state_dict[f'bert.encoder.layer.{i}.intermediate.dense.weight'])
        model.bert.encoder.layer[i].output.dense.w.load_state_dict(decomposed_state_dict[f'bert.encoder.layer.{i}.output.dense.weight'])

    trainer.train()
    t_result = trainer.evaluate()
    print('final performance:', t_result[f'eval_{metric_name}'])