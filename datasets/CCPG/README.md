# CCPG Dataset Setup

CCPG is a cloth-changing benchmark for person re-identification and gait
recognition.

Request access to the original dataset from the official CCPG provider.

The expected processed directory is:

```text
${data_root}/CCPG/CCPG-end2end-pkl-refined
```

If you start from the original silhouette and RGB folders, create the pickle
format with:

```shell
python datasets/CCPG/organize_ccpg.py \
  --sil_path /path/to/CCPG_G_SIL \
  --rgb_path /path/to/CCPG_D_MASK_FACE_SHOE \
  --output_path ${data_root}/CCPG/CCPG-end2end-pkl-refined
```

CCPG YAML configs:

```text
configs/ccpg/deepgaitv2_ccpg-ncb-pgfc-vm.yaml
configs/ccpg/gaitbase_ccpg-ncb-pgfc-vm.yaml
configs/ccpg/gaitpart_ccpg-ncb-vm.yaml
configs/ccpg/gaitset_ccpg-ncb-pgfc-vm.yaml
```

Run the main PGFC config directly:

```shell
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
  opengait/main.py \
  --cfgs configs/ccpg/deepgaitv2_ccpg-ncb-pgfc-vm.yaml \
  --phase train \
  --log_to_file
```
