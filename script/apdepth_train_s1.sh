BASE_DATA_DIR=${BASE_DATA_DIR:-"path/to/basedata"}
BASE_CKPT_DIR=${BASE_CKPT_DIR:-"path/to/pretrained_checkpoint"}

python apdepth_train_s1.py --config config/apdepth_train_s1.yaml \
    --base_data_dir $BASE_DATA_DIR \
    --base_ckpt_dir $BASE_CKPT_DIR \
    --output_dir output/stage1 \
    --no_wandb
