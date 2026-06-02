# PROTEAN: Prototype-based Federated Learning for Network Intrusion Detection

PROTEAN is a federated prototype-learning framework for network intrusion detection in settings with extreme class heterogeneity. Instead of aggregating only model parameters (as in FedAvg and similar methods), each participant computes compact, class-level prototypes in the model's embedding space and shares those summaries. These prototypes capture the behavioral signature of observed attack classes and are aggregated at a central server to form global class representations.



## Requirements

* Python 3.6+
* PyTorch 1.6+
* NumPy 1.18.5+
* scikit-learn
* tensorboardX
* tqdm
* pandas

## Datasets

| Dataset | Type | Description |
|---------|------|-------------|
| X-IIoTID | IoT network traffic | Multi-class intrusion detection |
| 5G-NIDD | 5G network traffic | Multi-class intrusion detection |

Place datasets under `data/` (or set `--data_dir`). 

Data is partitioned across clients using a **Dirichlet distribution** (controlled by `--alpha`), producing realistic non-IID splits where each client holds a skewed subset of traffic classes.

## Running Experiments

All experiments are launched from `exps/`.

### Basic federated run (Dirichlet non-IID)

```bash
python federated_main.py --dataset xiiotid --num_classes 9 \
  --num_users 10 --rounds 10 --alpha 0.25 \
  --train_ep 3 
```

### With differential privacy

```bash
python federated_main.py --dataset xiiotid --num_classes 9 \
  --num_users 10 --rounds 10 --alpha 0.25 \
  --dp True --epsilon 5 --delta 1e-3 
```

## Options

### Federated Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--rounds` | 10 | Number of communication rounds |
| `--num_users` | 10 | Number of clients |
| `--train_ep` | 3 | Local training episodes per round |
| `--local_bs` | 64 | Local batch size |
| `--mode` | `task_heter` | `task_heter` or `model_heter` |
| `--ld` | 1 | Weight of prototype loss |
| `--alg` | `fedproto` | Federated algorithm |


### Data Partitioning

| Argument | Default | Description |
|----------|---------|-------------|
| `--alpha` | 0.25 | Dirichlet concentration — lower = more heterogeneous |
| `--iid` | 0 | Set to 1 for IID splits |

### Training Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` | 0.01 | Learning rate |
| `--optimizer` | `adam` | Optimizer type |
| `--momentum` | 0.5 | SGD momentum |
| `--seed` | 42 | Random seed |


### Differential Privacy

| Argument | Default | Description |
|----------|---------|-------------|
| `--dp` | False | Enable differential privacy |
| `--epsilon` | 5 | Privacy budget ε |
| `--delta` | 1e-3 | Privacy budget δ |
