#!/usr/bin/env bash
set -e
set -x

ckpt_dir=${ckpt_dir:-checkpoints}
target_dir=${ckpt_dir}/ApDepth

repo_id="developy/ApDepth"

mkdir -p $ckpt_dir

if [ -d "$target_dir" ]; then
    echo "Checkpoint already exists at $target_dir, skip download."
    exit 0
fi

pip install -q huggingface_hub

huggingface-cli download $repo_id \
    --local-dir $target_dir \
    --local-dir-use-symlinks False

echo "Model downloaded to $target_dir"