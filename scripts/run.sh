#!/bin/bash
# Run from project root: bash scripts/run.sh

echo "Script: $0"

# Basic federated run — X-IIoTID, Dirichlet non-IID (alpha=0.25)
python exps/federated_main.py --dataset xiiotid \
  --num_users 10 --rounds 100 --alpha 0.25 \
  --dirichlet 1 --train_ep 3 --local_bs 64 --ld 1

# Basic federated run — 5G-NIDD, Dirichlet non-IID (alpha=0.25)
python exps/federated_main.py --dataset 5gnidd \
  --num_users 10 --rounds 100 --alpha 0.25 \
  --dirichlet 1 --train_ep 3 --local_bs 64 --ld 1

# With differential privacy — X-IIoTID
python exps/federated_main.py --dataset xiiotid \
  --num_users 10 --rounds 100 --alpha 0.25 \
  --dirichlet 1 --train_ep 3 --local_bs 64 --ld 1 \
  --dp True --epsilon 5 --delta 1e-3 --clip_threshold 100.0
