#!/bin/bash
set -e
export PATH=/home/qqelick/hibana/.venv/bin:$PATH
cd /home/qqelick/mtp_zh
exec /home/qqelick/hibana/.venv/bin/python training/train_mtp_kl.py   --data-dir /home/qqelick/mtp_zh --out-dir /home/qqelick/mtp_zh/out2   --epochs 2 --lr 6e-6 --batch-size 2 --grad-accum 4 --num-anchors 64 --lora-rank 32
