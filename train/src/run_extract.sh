#!/bin/bash
set -e
export PATH=/home/qqelick/hibana/.venv/bin:$PATH
export HF_HOME=/home/qqelick/.cache/huggingface
export HIBANA_HF_CACHE=/home/qqelick/.cache/huggingface
export HIBANA_W3_CHECKPOINT=/home/qqelick/models/w3-flashnext
export HIBANA_W3_DECODE=1
export HIBANA_VISION=0
export HIBANA_FLASHNEXT_EXPERTS=w3
cd /home/qqelick/mtp_zh
exec /home/qqelick/hibana/.venv/bin/python training/extract_mtp_features.py   --hibana-dir /home/qqelick/hibana   --corpus /home/qqelick/mtp_zh/corpus_zh_all.jsonl   --out-dir /home/qqelick/mtp_zh --shard-size 60
