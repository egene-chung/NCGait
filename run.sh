#!/usr/bin/env bash
set -euo pipefail

export PYTHONHASHSEED=0

# deepgaitv2 ccpg
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=12389 opengait/main.py --cfgs ./configs/ccpg/deepgaitv2_ccpg-ncb-pgfc-vm.yaml --phase train --log_to_file
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=12389 opengait/main.py --cfgs ./configs/ccpg/deepgaitv2_ccpg-ncb-pgfc-vm.yaml --phase test --log_to_file
