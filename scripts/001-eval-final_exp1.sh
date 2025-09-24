export CUDA_HOME=/usr/local/cuda-11.1/

# CUDA_VISIBLE_DEVICES=7 python tools/train_val.py --config configs/60-06-04-stereodetrlight_w_uncertainty_flip_crop_depthreg.yaml

# CUDA_VISIBLE_DEVICES=2 python tools/train_val.py --config configs/70-03-01-disp_s19.yaml
# 
CUDA_VISIBLE_DEVICES=0 python tools/train_val.py --config /server19/mushiyi/smb9_msy/02-Code/05-stereo-detr/StereoDETR-Open/configs/001-final_exp1.yaml -e

