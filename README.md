<div align="center">
  <h1><strong>ApDepth</strong></h1>
  <h3><strong>Aiming for Precise Monocular Depth Estimation Based on Diffusion Models</strong></h3>
</div>

<p align="center">
  <a href="#news"><img src="doc/badges/v21-version.svg" alt="ApDepth V2.1"></a>&nbsp;
  <a href="https://haruko386.github.io/research"><img src="doc/badges/v21-website.svg" alt="Project website"></a>&nbsp;
  <a href="#citation"><img src="doc/badges/v21-paper.svg" alt="Paper and citation"></a>&nbsp;
  <a href="https://huggingface.co/spaces/developy/ApDepth"><img src="doc/badges/v21-demo.svg" alt="Live demo"></a>&nbsp;
  <a href="https://huggingface.co/developy/ApDepth"><img src="doc/badges/v21-weights.svg" alt="Model weights"></a>&nbsp;
  <a href="#training"><img src="doc/badges/v21-training.svg" alt="Training recipes"></a>
</p>

---

![ApDepth monocular depth estimation examples](doc/cover.png)

## Overview

**ApDepth** turns Stable Diffusion 2.1 into a deterministic, single-step
monocular depth estimator. It combines a frozen Depth Anything V2 prior with a
fine-tuned diffusion U-Net to recover accurate scene geometry and crisp object
boundaries without iterative denoising.

> [!IMPORTANT]
>
> ApDepth builds on [**Marigold**](https://marigoldmonodepth.github.io), the
> CVPR 2024 Best Paper [*Repurposing Diffusion-Based Image Generators for
> Monocular Depth Estimation*](https://arxiv.org/abs/2312.02145).

## News

- **2026-09-22:** Preparing **ApDepth V2-1** with direct SD2.1 fine-tuning,
  SDWT and optional FFT refinement.
- **2026-04-06:** `ApDepth V2-0` is released!
- **2026-04-03:** We officially release the code for **ApDepth**! Stage 1 feature-alignment training is maintained in [ApDepth_Stage1](https://github.com/Haruko386/ApDepth_Stage1); this repository provides Stage 2 training, inference and evaluation.
- **2026-01-15:** We successfully introduce a spatial-preserving **Conv Adapter** and a **Cosine Similarity Loss** to enhance feature alignment, alongside a **Pixel-level $L_1$ Loss** to establish an accurate global metric scale.
- **2025-10-25:** Inspired by DepthMaster, we propose a two-stage loss function training strategy based on `ApDepth V1-0`. In the first stage, we perform foundational training using MSE loss. In the second stage, we learn edge structures through FFT loss. Based on this, we introduce `ApDepth V1-1`.
- **2025-10-09:** We propose a novel diffusion-based depth estimation framework guided by pre-trained models.
- **2025-09-23:** We change Marigold from **Stochastic multi-step generation** to **Deterministic one-step perception**.
- **2025-08-10:** Trying to make some optimizations in Feature Expression.
- **2025-05-08:** Clone `Marigold` to local.

## Quick start

Try ApDepth in the [live demo](https://huggingface.co/spaces/developy/ApDepth),
browse the [project gallery](https://haruko386.github.io/research), or run it
locally using the instructions below.

### Requirements

The model was trained on:

- Ubuntu 22.04 LTS, Python 3.12.9,  CUDA 11.8, `NVIDIA RTX 6000 Ada Generation`

Inference was tested on:

- Ubuntu 22.04 LTS, Python 3.12.9,  CUDA 11.8, `NVIDIA GeForce RTX 4090`

#### Windows

We recommend running the code in WSL2:

1. Install WSL following [installation guide](https://learn.microsoft.com/en-us/windows/wsl/install#install-wsl-command).
1. Install CUDA support for WSL following [installation guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#cuda-support-for-wsl-2).
1. Find your drives in `/mnt/<drive letter>/`; check [WSL FAQ](https://learn.microsoft.com/en-us/windows/wsl/faq#how-do-i-access-my-c--drive-) for more details. Navigate to the working directory of choice. 

### Installation

Clone the repository (requires git):

```bash
git clone https://github.com/Haruko386/ApDepth.git
cd ApDepth
```

#### Conda

Create and activate the project environment:

```bash
conda create -n apdepth python==3.12.9
conda activate apdepth
pip install -r requirements.txt
```

> [!NOTE]
>
> Keep the environment activated before running the inference script. 
> Activate the environment again after restarting the terminal session.

### Docker

For a streamlined setup, we provide a Docker environment that pre-installs all necessary dependencies, including PyTorch, CUDA, and evaluation tools.

**1. Build the Docker Image**

Ensure you have Docker installed. Run the following command in the root directory of the repository:

```bash
docker build -t apdepth:latest .
```

**2. Run the Container**

To utilize GPU acceleration, ensure the NVIDIA Container Toolkit is installed. We recommend mounting your local input and output directories to easily access your inference results:

```bash
docker run --gpus all -it --rm \
    -v $(pwd)/input:/workspace/ApDepth/input \
    -v $(pwd)/output:/workspace/ApDepth/output \
    apdepth:latest
```

Inside the container, the `apdepth` Conda environment is activated automatically.

## Inference

### Prepare images

1. Use selected images under `input`

1. Or place your images in a directory, for example, under `input/test-image`, and run the following inference command.

### Paper setting

This setting corresponds to our paper. For academic comparison, please run with this setting.

```bash
python run.py \
    --checkpoint checkpoint/ApDepth \
    --ensemble_size 1 \
    --processing_res 0 \
    --input_rgb_dir input/example-1 \
    --output_dir output/example-1
```

You can find all results in `output/example-1`. Enjoy!

### Inference options

The default settings are optimized for the best result. However, the behavior of the code can be customized:

- Trade-offs between the **accuracy** and **speed** (for both options, larger values result in better accuracy at the cost of slower inference.)
  - `--ensemble_size`: Number of inference passes in the ensemble. 
  - `--processing_res`: the processing resolution; set as 0 to process the input resolution directly. When unassigned (`None`), will read default setting from model config. Default: ~~768~~ `None`.
  - `--output_processing_res`: produce output at the processing resolution instead of upsampling it to the input resolution. Default: False.
  - `--resample_method`: the resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic`, or `nearest`. Default: `bilinear`.

- `--half_precision` or `--fp16`: Run with half-precision (16-bit float) to have faster speed and reduced VRAM usage, but might lead to suboptimal results.
- `--seed`: Random seed can be set to ensure additional reproducibility. Default: None (unseeded). Note: forcing `--batch_size 1` helps to increase reproducibility. To ensure full reproducibility, [deterministic mode](https://pytorch.org/docs/stable/notes/randomness.html#avoiding-nondeterministic-algorithms) needs to be used.
- `--batch_size`: Batch size of repeated inference. Default: 0 (best value determined automatically).
- `--color_map`: [Colormap](https://matplotlib.org/stable/users/explain/colors/colormaps.html) used to colorize the depth prediction. Default: Spectral. Set to `None` to skip colored depth map generation.
- `--apple_silicon`: Use Apple Silicon MPS acceleration.

## Evaluation

Install additional dependencies:

```bash
pip install -r requirements+.txt -r requirements.txt
```

Set data directory variable (also needed in evaluation scripts) and download [evaluation datasets](https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset) into corresponding subfolders:

```bash
export BASE_DATA_DIR=<YOUR_DATA_DIR>  # Set target data directory

wget -r -np -nH --cut-dirs=4 -R "index.html*" -P ${BASE_DATA_DIR} https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/
```

Run inference and evaluation scripts, for example:

```bash
# Run inference
bash script/eval/11_infer_nyu.sh

# Evaluate predictions
bash script/eval/12_eval_nyu.sh
```

Alternatively, use the following script to evaluate all datasets.

```bash
# Evaluate all datasets
bash script/eval/00_test_all.sh
```
You can get the result under `output/eval`

> [!IMPORTANT]
>
> Although the seed has been set, the results might still be slightly different on different hardware.



## Training

Three training methods are retained. The mixed SDWT + FFT method is the
recommended recipe; the SDWT-only method is kept for controlled comparison, and the
original Stage 1 + legacy FFT workflow remains reproducible.

| Method | Initialization | Objective | Config |
| --- | --- | --- | --- |
| 1. Mixed SDWT + FFT (recommended) | Original SD2.1 | MSE + pixel L1 + gradient, then gradual SDWT + low-weight FFT | `config/train_sd2_sdwt_fft.yaml` |
| 2. SDWT only | Original SD2.1 | MSE + pixel L1 + gradient, then gradual SDWT | `config/train_sd2_sdwt.yaml` |
| 3. Legacy ApDepth/FFT | Stage 1 U-Net checkpoint | Reconstruction, then the original latent FFT transition | `config/train_apdepth.yaml` |

Methods 1 and 2 are single-run Stage 2 training methods and do not execute or
load Stage 1. Their implementation details and ablations are documented
[here](doc/sd2_sdwt_training.md).

Based on the previously created environment, install extended requirements:

```bash
pip install -r requirements++.txt -r requirements+.txt -r requirements.txt
```

Set environment parameters for the data directory:

```bash
export BASE_DATA_DIR=YOUR_DATA_DIR  # directory of training data
export BASE_CKPT_DIR=YOUR_CHECKPOINT_DIR  # directory of pretrained checkpoint
```

Download Stable Diffusion v2 [checkpoint](https://huggingface.co/sd2-community/stable-diffusion-2-1) into `${BASE_CKPT_DIR}`

Download the ViT-G checkpoint of [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) to `DA2/checkpoints/depth_anything_v2_vitg.pth`. The ApDepth pipeline uses it to generate the depth prior during training and inference.

Prepare [Hypersim](https://github.com/apple/ml-hypersim) and [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/) under `${BASE_DATA_DIR}`. Please refer to [this README](script/dataset_preprocess/hypersim/README.md) for Hypersim preprocessing. Configure the training paths in `config/dataset/dataset_train.yaml` and prepare the validation datasets listed in `config/dataset/dataset_val.yaml` and `config/dataset/dataset_vis.yaml`.

------------

### Method 1: direct SD2 training with SDWT + FFT (recommended)

This method starts from the original SD2.1 weights. During iterations 0-8,000,
it learns the basic depth mapping with latent MSE, pixel L1 and gradient loss.
From iteration 8,000 to 12,000, SDWT increases from 0 to 1.0 while latent
FFT increases from 0 to 0.2. The reconstruction losses remain active for the
rest of training, and the VAE decoder is fine-tuned from iteration 8,000.

SDWT matches local depth distributions with an explicit pixel-displacement
cost. It subtracts entropic self-costs, evaluates both regular and half-window
shifted grids, and weights each window by its valid support. These changes keep
limited tolerance to noisy boundary labels without making the loss invariant to
arbitrary permutations inside a window.

```bash
python train.py --config config/train_sd2_sdwt_fft.yaml --no_wandb
```

Do not pass `--init_checkpoint`. Resume an interrupted run with:

```bash
python train.py \
    --resume_run output/train_sd2_sdwt_fft/checkpoint/latest --no_wandb
```

### Method 2: direct SD2 training with SDWT only

This method uses the same original SD2.1 initialization, reconstruction-loss
schedule and VAE decoder adaptation as Method 1, but leaves latent FFT disabled.
It is the direct comparison for measuring the contribution of FFT.

```bash
python train.py --config config/train_sd2_sdwt.yaml --no_wandb
```

Do not pass `--init_checkpoint`. Resume an interrupted run with:

```bash
python train.py \
    --resume_run output/train_sd2_sdwt/checkpoint/latest --no_wandb
```

For both direct-SD2 methods, `BASE_CKPT_DIR/stable-diffusion-2-1` must contain
the original SD2.1 Diffusers pipeline. Their checkpoints save both the U-Net and
the fine-tuned VAE.

### Method 3: legacy Stage 1 + ApDepth/FFT training (ApDepth v2-0)

First complete the feature-alignment Stage 1 training. Stage 1 is maintained
separately in
[Haruko386/ApDepth_Stage1](https://github.com/Haruko386/ApDepth_Stage1).
Follow that repository's setup and training instructions. This repository contains
the legacy Stage 2 trainer, but does not include Stage 1 code or dependencies.

Then initialize the legacy Stage 2 from the Stage 1 checkpoint with
`--init_checkpoint`. This loads the U-Net and a saved VAE if present; optimizer and iteration counters
start fresh using the Stage 2 config. `config/train_apdepth.yaml` keeps far-depth
supervision disabled throughout the main training stage:

```bash
python train.py --config config/train_apdepth.yaml \
    --init_checkpoint /path/to/stage1/checkpoint/iter_020000 --no_wandb
```

The initialization directory must contain
`unet/diffusion_pytorch_model.safetensors`. `--init_checkpoint` rejects `.bin`
checkpoints.

The transition to latent frequency loss is controlled by
`latent_freq_loss.gradual_transition` in `config/train_apdepth.yaml`. When `true`,
the reconstruction loss weight decreases linearly from 1 to 0 between iteration
20,000 and `max_iter`, while the frequency-loss weight increases from 0 to 1.
When `false` or omitted, training preserves the original hard switch immediately
after iteration 20,000. TensorBoard records the active frequency weight as
`train/freq_loss_weight`.

Resume from a checkpoint, e.g.

```bash
python train.py --resume_run output/train_apdepth/checkpoint/latest --no_wandb
```

### Inference after Method 1 or Method 2

Direct-SD2 checkpoints are training-component checkpoints, so inference combines
the original SD2.1 pipeline with the saved U-Net and VAE. For Method 1:

```bash
python run.py --checkpoint "${BASE_CKPT_DIR}/stable-diffusion-2-1" \
    --training_checkpoint output/train_sd2_sdwt_fft/checkpoint/iter_021000 \
    --input_rgb_dir input/example-1 --output_dir output/sd2_sdwt_fft \
    --ensemble_size 1 --processing_res 0
```

For Method 2, replace `train_sd2_sdwt_fft` with `train_sd2_sdwt` in the checkpoint
path. `infer.py` also accepts `--training_checkpoint` for dataset evaluation.

------------

## Optional post-training after Stage 2

**Start this step only after Stage 2 training has finished.** The new VKITTI
far-depth supervision feature is enabled by `config/train_sky_finetune.yaml` for
a separate post-training run initialized from a completed checkpoint from any
of the three methods. For Methods 1 and 2, the saved VAE is loaded and retained
together with the U-Net.

Measured depths in [80, 655.35] m supply an additional clipped far-depth target
before VAE encoding; missing depth remains excluded. Evaluation masks and
normalization quantiles are unchanged. The separately averaged far latent MSE
and pixel L1 terms have weight 0.5. For a post-training ablation, set
`far_depth_supervision.enabled: false` in the post-training config.

Use the same environment, datasets and base checkpoints prepared for Stage 2.
First audit the native VKITTI depth and target coverage:

```bash
python -m script.audit_far_supervision --config config/train_sky_finetune.yaml \
    --base_data_dir "${BASE_DATA_DIR}" --samples 20
```

Check the RGB, far masks, target images and `coverage.json` in `output/far_audit`.
If preprocessing replaced the far-plane values with zero, restore the original
depth files. Zero depth is not treated as sky. Adjust each dataset config's `dir`
to match the actual layout under `BASE_DATA_DIR`.

Select the corresponding completed checkpoint:

| Method | Example checkpoint |
| --- | --- |
| 1. SDWT + FFT | `output/train_sd2_sdwt_fft/checkpoint/iter_021000` |
| 2. SDWT only | `output/train_sd2_sdwt/checkpoint/iter_021000` |
| 3. Legacy ApDepth/FFT | `output/train_apdepth/checkpoint/iter_021000` |

Then start a new post-training run, adjusting the path if needed:

```bash
python train.py --config config/train_sky_finetune.yaml \
    --init_checkpoint /path/to/completed/checkpoint/iter_021000 --no_wandb
```

This resets the optimizer, learning-rate schedule and iteration counter, then
runs 6000 new optimization steps at LR 5e-6 with 100 warmup steps and saves
backups every 1000 steps in `output/train_sky_finetune/checkpoint`. It retains the
90% Hypersim / 10% VKITTI sampling, original batch settings and periodic validation.
These 6000 steps use latent MSE + pixel L1 reconstruction plus the new far-depth
loss; the selected Stage 2 objective has already finished. Initialization requires
`unet/diffusion_pytorch_model.safetensors` in the supplied checkpoint directory.

If this post-training run is interrupted, resume its own checkpoint:

```bash
python train.py --resume_run output/train_sky_finetune/checkpoint/latest --no_wandb
```

`--resume_run` restores the saved config and training state. Starting post-training
requires `--init_checkpoint` with the post-training config above; resuming a
Stage 2 checkpoint would continue Stage 2 with its saved settings.
`--resume_run` and `--init_checkpoint` cannot be combined.

Monitor `train/far_loss`, `train/far_pixel_ratio` and `train/far_latent_ratio`.
Compare KITTI and NYU against the completed Stage 2 model after post-training; this change
alone does not establish a metric improvement.

## Evaluating trained checkpoints

The new direct-SD2 recipe saves both U-Net and VAE; use `--training_checkpoint`
as shown above to load them together. The following instructions apply to legacy
U-Net-only checkpoints.

Legacy Stage 2 and optional post-training update and save the U-Net. For inference, copy a complete
ApDepth pipeline checkpoint to a new directory, then replace that copy's `unet/`
folder with the selected training checkpoint's `unet/`. The pipeline's VAE,
tokenizer, text encoder and scheduler are still required; a training checkpoint
alone is not a complete inference pipeline. Then refer to [evaluation](#evaluation).

> [!IMPORTANT]
>
> Although random seeds have been set, the training result might be slightly different on different hardwares. It's recommended to train without interruption.



## Contributing

Please refer to [this](CONTRIBUTING.md) instruction.

## Troubleshooting

| Problem                                                                                                                     | Solution                                                                  |
| --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| (Windows) Invalid DOS bash script on WSL / `$'\r': command not found` / `set: invalid option`                               | Run `dos2unix <script_name>` to convert script format                     |
| (Windows) Multiple `.sh` scripts fail due to CRLF line endings                                                              | Run `find . -name "*.sh" -exec dos2unix {} +` to fix all scripts          |
| (Windows) error on WSL: `Could not load library libcudnn_cnn_infer.so.8. Error: libcuda.so: cannot open shared object file` | Run `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH`            |
| HuggingFace model download incomplete / corrupted                                                                           | Re-run with `--resume-download` or ensure stable network                  |
| `model_index.json not found` when loading checkpoint                                                                        | Ensure the model is fully downloaded and placed at `checkpoints/ApDepth/` |
| Dataset loading error: `tarfile.ReadError: unexpected end of data`                                                          | Re-download dataset; the `.tar` file is likely corrupted or incomplete    |



## Citation
Please cite our paper:

```bibtex
@InProceedings{wang26apdepth,
      title={ApDepth: Aiming for Precise Monocular Depth Estimation Based on Diffusion Models},
      author={Jiawei Wang, Mingbo Lei, Haoze Shou, Yusu Liang and Yuan Shuai},
      booktitle = {Arxiv},
      year={2026}
}
```

## License

This work is licensed under the Apache License, Version 2.0 (as defined in the [LICENSE](LICENSE.txt)).

By downloading and using the code and model you agree to the terms in the [LICENSE](LICENSE.txt).
