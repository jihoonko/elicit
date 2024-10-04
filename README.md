# ELiCiT: Effective and Lightweight Lossy Compression of Tensors

This repository contains the official implementation of ELiCiT, described in the paper ELiCiT: Effective and Lightweight Lossy Compression of Tensors, Jihoon Ko, Taehyung Kwon, Jinhong Jung, and Kijung Shin, ICDM 2024 (To be appeared).

ELiCiT (**E**ffective and **L**ightweight Lossy **C**ompress**i**on of **T**ensors) is an algorithm for lossy compression of tensors. ELiCiT has the following advantages:

- **Compact and Accurate:** It consistently achieves a better trade-off between compressed size and approximation error than all considered competitors. Specifically, ELiCiT compresses tensors to sizes 1.51-5.05x smaller than competitors while achieving similar fitness. It also achieves 5-48% better fitness than competitors with similar output sizes.
- **Fast:** While giving similar outputs with better fitness, ELiCiT is 11.8-96.0x faster than deep-learning methods.
- **Applicable:** It is successfully applied to matrix completion and neural network compression, providing a better trade-off between model size and application performance, compared to state-of-the-art competitors for these applications.


**Note: Refer to `supplementary_material.pdf` for the paper appendix.**

## Requirements

The common required Python packages are listed as follows:

- pytorch ≥ 1.8.1
- torch-scatter ≥ 2.0.8

For tensor completion, you need to install the following packages additionally:

- optuna ≥ 3.1.1

For neural-network compression, you need to install the following packages additionally:

- transformers==4.30.2
- datasets==2.13.1
- evaluate==0.4.0
- accelerate==0.20.3
- seqeval==1.2.2

## Training & Evaluation

### Tensor Compression (in `tensor_compression/` directory)

To train and evaluate a qELiCiT model, run this command:

```bash
python train.py --input-path [filename] --output-path [filename] --num-features [num-features] --lr1 [lr1] --lr2 [lr2] --gpus [gpu-ids] --seed [seed]
```

For example, you can compress the tensor stored in `tensor_compression/uber.npy` with qELiCiT using the following command:

```bash
python train.py --input-path uber.npy --output-path uber_40.elicit --num-features 40 --lr1 1e-3 --lr2 1e-2 --gpus 0 1 2 3 --seed 0
```

To measure the fitness of the compressed output, run this command: 

```bash
python evaluate.py --original-path [original-tensor-path] --compressed-path [compressed-output-path] --gpus [gpu-ids]
```

In the directory, we provide a compressed output `tensor_compression/example_uber_30.elicit` of the Uber dataset. You can measure the fitness using the following command:

```bash
python evaluate.py --original-path uber.npy --compressed-path example_uber_40.elicit --gpus 0,1,2,3
```

### Matrix Completion (in `matrix_completion/` directory)

To find the optimal hyperparameter settings of qELiCiT++, run this command:

```bash
python search.py --dataset-name [ml-10m|ml-1m|ml-100k|douban|flixster] --budget [4|8|16|32] --gpu [gpu-id]
```

For example, you can find the optimal hyperparameter settings of qELiCiT++ with budget 32*(#rows + #columns) on the douban dataset using the following command:

```bash
python search.py --dataset-name douban --budget 32 --gpu 0
```

To evaluate the performance of qELiCiT++, run this command:

```bash
python evaluate.py --dataset-name [ml-10m|ml-1m|ml-100k|douban|flixster] --budget [4|8|16|32] --lamb1 [lamb1] --lamb2 [lamb2] --lamb3 [lambda3] --lr1 [lr1] --lr2 [lr2] --gpu [gpu-id]
```

### Neural-network Compression (in `neuralnet_compression/` directory)

To fine-tune the BERT_base model on the GLUE subtasks, run this command:

```bash
python finetune_bert.py --task [cola|mnli|mrpc|qnli|qqp|sst2|stsb] --lr [2e-5|3e-5|5e-5] --weight-decay [0|0.01] --num-epochs [2|3|4] --seed [0|1000|2000|3000|4000] --gpu [gpu-id]
```

To compress the fine-tuned model using TFW-qELiCiT, run this command:

```bash
python tfw_qelicit.py --task [cola|mnli|mrpc|qnli|qqp|sst2|stsb] --lr [2e-5|3e-5|5e-5] --weight-decay [0|0.01] --num-epochs [2|3|4] --seed [0|1000|2000|3000|4000] --gpu [gpu-id]
```

To fine-tune the compressed model, run this command:

```bash
python finetune_ours.py --task [cola|mnli|mrpc|qnli|qqp|sst2|stsb] --target [target-checkpoint-path] --lr [2e-5|3e-5|5e-5] --weight-decay [0|0.01] --num-epochs [2|3|4] --seed [0|1000|2000|3000|4000] --gpu [gpu-id]
```

## Datasets

### Tensor Compression

We used 8 real-world datasets, which are listed below. All datasets we considered are available at the [TensorCodec](https://github.com/kbrother/TensorCodec) repository.

| Order | Name | Shape | #Entries | Brief description |
| --- | --- | --- | --- | --- |
| 4 | Absorb | 192 x 288 x 30 x 120 | 199.1M | Climate |
|  | NYC | 265 x 265 x 28 x 35 | 68.8M | Traffic volume |
| 3 | Action | 100 x 570 x 567 | 32.3M | Video features |
|  | Activity | 337 x 570 x 320 | 61.5M | Video features |
|  | Airquality | 5600 x 362 x 6 | 12.2M | Climate |
|  | PEMS | 963 x 144 x 440 | 61.0M | Traffic volume |
|  | Stock | 1317 x 88 x 916 | 106.2M | Stock |
|  | Uber | 183 x 24 x 1140 | 5.0M | Traffic volume |

### Matrix Completion

We used 5 real-world matrices containing ratings of movies provided by users. For the douban and flixster datasets, we provide the resources in the `matrix_completion/data/` directory. The ML-100K, ML-1M and ML-10M datasets are available at https://grouplens.org/datasets/movielens/, and we provide `matrix_completion/data/split.py` for splitting the ratings into train/val/test split.

### Neural-network Compression

We used 7 downstream tasks from the GLUE benchmark for evaluating our method and its competitors. The datasets for the tasks will be automatically downloaded when you run the provided code for training and evaluation.


## Terms and Conditions
If you use this code as part of any published research, please consider acknowledging our ICDM 2024 paper.

```
@inproceedings{ko2024elicit,
  title={ELiCiT: Effective and Lightweight Lossy Compression of Tensors},
  author={Ko, Jihoon and Kwon, Taeyhung and Jung, Jinhong and Shin, Kijung},
  booktitle={2024 IEEE International Conference on Data Mining (ICDM)},
  year={2024},
}
```
