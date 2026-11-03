# SUSTech1K Dataset Setup

Obtain SUSTech1K from the official dataset provider.

After extracting the official pickle release, place it under `data_root`
(set in `env.yaml`):

```text
${data_root}/SUSTech1K/SUSTech1K-Released-pkl
```

The expected structure is:

```text
SUSTech1K-Released-pkl/
|-- 0000/
|   |-- 00-nm/
|   |   |-- 000/
|   |   |   |-- 00-000-LiDAR-PCDs.pkl
|   |   |   |-- 01-000-LiDAR-PCDs_depths.pkl
|   |   |   |-- 02-000-LiDAR-PCDs_sils.pkl
|   |   |   |-- 03-000-Camera-Pose.pkl
|   |   |   |-- 04-000-Camera-Ratios-HW.pkl
|   |   |   |-- 05-000-Camera-RGB_raw.pkl
|   |   |   |-- 06-000-Camera-Sils_aligned.pkl
|   |   |   |-- 07-000-Camera-Sils_raw.pkl
|   |   |   |-- 08-sync-000-LiDAR-PCDs.pkl
|   |   |   |-- 09-sync-000-LiDAR-PCDs_depths.pkl
|   |   |   |-- 10-sync-000-LiDAR-PCDs_sils.pkl
|   |   |   |-- 11-sync-000-Camera-Pose.pkl
|   |   |   |-- 12-sync-000-Camera-Ratios-HW.pkl
|   |   |   |-- 13-sync-000-Camera-RGB_raw.pkl
|   |   |   |-- 14-sync-000-Camera-Sils_aligned.pkl
|   |   |   |-- 15-sync-000-Camera-Sils_raw.pkl
```

The configs in this repository use `06-000-Camera-Sils_aligned.pkl` through the
`data_in_use` field.

SUSTech1K YAML configs:

```text
configs/sustech1k/deepgaitv2_sustech1k-ncb-pgfc-vm.yaml
configs/sustech1k/gaitbase_sustech1k-ncb-pgfc-vm.yaml
configs/sustech1k/gaitpart_sustech1k-ncb-vm.yaml
configs/sustech1k/gaitset_sustech1k-ncb-pgfc-vm.yaml
```

Run the main PGFC config directly:

```shell
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
  opengait/main.py \
  --cfgs configs/sustech1k/deepgaitv2_sustech1k-ncb-pgfc-vm.yaml \
  --phase train \
  --log_to_file
```
