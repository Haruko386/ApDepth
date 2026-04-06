# ApDepth: Aiming for Precise Monocular Depth Estimation Based on Diffusion Models

This repository is based on [Marigold](https://marigoldmonodepth.github.io), CVPR 2024 Best Paper: [**Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation**](https://arxiv.org/abs/2312.02145)

[![Website](doc/badges/badge-website.svg)](https://haruko386.github.io/research)
[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-green)](https://huggingface.co/developy/ApDepth)
[![Hugging Face Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-purple)](https://huggingface.co/spaces/developy/ApDepth)

[**Haruko386**](https://haruko386.github.io/),
[Shuai Yuan](https://syjz.teacher.360eol.com/teacherBasic/preview?teacherId=23776),
[Mingbo Lei](https://github.com/Ltohka), 
[Yibo Chen](#)

![cover](doc/cover.png)

> We present **ApDepth**, a deterministic single-step diffusion framework for monocular depth estimation. Built upon Marigold (derived from Stable Diffusion) and fine-tuned on synthetic datasets (Hypersim and Virtual KITTI), ApDepth overcomes the feature representation bottlenecks of standard diffusion models. Ultimately, our framework achieves **highly competitive geometric accuracy** and **exceptional object edge refinement**, all while delivering significantly accelerated inference speeds.

## 📢 News
- **2026-04-06:** `ApDepth V2-0` is released!
- **2026-04-03:** We officially release the complete code for **ApDepth**! The repository now includes the full coarse-to-fine two-stage training pipeline and evaluation scripts.
- **2026-01-15:** We successfully introduce a spatial-preserving **Conv Adapter** and a **Cosine Similarity Loss** to enhance feature alignment, alongside a **Pixel-level $L_1$ Loss** to establish an accurate global metric scale.
- **2025-10-25:** Inspired by DepthMaster, we propose a two-stage loss function training strategy based on `ApDepth V1-0`. In the first stage, we perform foundational training using MSE loss. In the second stage, we learn edge structures through FFT loss. Based on this, we introduce `ApDepth V1-1`.
- **2025-10-09:** We propose a novel diffusion-based depth estimation framework guided by pre-trained models.
- **2025-09-23:** We change Marigold from **Stochastic multi-step generation** to **Deterministic one-step perception**.
- **2025-08-10:** Trying to make some optimizations in Feature Expression.
- **2025-05-08:** Clone `Marigold` to local.

## 🚀 Usage

**We offer several ways to interact with ApDepth**:

1. A free online interactive demo is available here: <a href="https://huggingface.co/spaces/developy/ApDepth"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-purple" height="18"></a>

2. If you just want to see the examples, visit our gallery: <a href="https://haruko386.github.io/research"><img src="doc/badges/badge-website.svg" height="16"></a>

3. Local development instructions with this codebase are given below.

## 🛠️ Setup
The Model was trained on:

- Ubuntu 22.04 LTS, Python 3.12.9,  CUDA 11.8, `NVIDIA RTX 6000 Ada Generation`

The inference code was tested on:

- Ubuntu 22.04 LTS, Python 3.12.9,  CUDA 11.8, `NVIDIA GeForce RTX 4090`

### 🪧 A Note for Windows users

We recommend running the code in WSL2:

1. Install WSL following [installation guide](https://learn.microsoft.com/en-us/windows/wsl/install#install-wsl-command).
1. Install CUDA support for WSL following [installation guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#cuda-support-for-wsl-2).
1. Find your drives in `/mnt/<drive letter>/`; check [WSL FAQ](https://learn.microsoft.com/en-us/windows/wsl/faq#how-do-i-access-my-c--drive-) for more details. Navigate to the working directory of choice. 

### 📦 Repository

Clone the repository (requires git):

```bash
git clone https://github.com/Haruko386/ApDepth.git
cd ApDepth
```

### 💻 Dependencies

 **Using Conda:** 
    Alternatively, create a Python native virtual environment and install dependencies into it:

```bash
conda create -n apdepth python==3.12.9
conda activate apdepth
pip install -r requirements.txt
```

> [!NOTE]
>
> Keep the environment activated before running the inference script. 
> Activate the environment again after restarting the terminal session.

### 🐳 Docker Setup

For a streamlined setup, we provide a Docker environment that pre-installs all necessary dependencies, including PyTorch, CUDA, and evaluation tools.

**1. Build the Docker Image**

Ensure you have Docker installed. Run the following command in the root directory of the repository:

```bash
docker build -t apdepth:latest .
```

**2. Run the Container**

To utilize GPU acceleration, ensure the NVIDIA Container Toolkit is installed. We recommend mounting your local input and output directories to easily access your inference results:

```Bash
docker run --gpus all -it --rm \
    -v $(pwd)/input:/workspace/ApDepth/input \
    -v $(pwd)/output:/workspace/ApDepth/output \
    apdepth:latest
Once inside the container, the apdepth conda environment is activated by default, and you can directly execute the inference or training scripts.
```
## 🏃 Testing on your images

### 📷 Prepare images

1. Use selected images under `input`

1. Or place your images in a directory, for example, under `input/test-image`, and run the following inference command.

### 🎮 Run inference with paper setting

This setting corresponds to our paper. For academic comparison, please run with this setting.

```bash
python run.py \
    --checkpoint checkpoints/ApDepth \
    --ensemble_size 1 \
    --processing_res 0 \
    --input_rgb_dir input/example-1 \
    --output_dir output/example-1
```

You can find all results in `output/example-1`. Enjoy!

### ⚙️ Inference settings

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

## 🦿 Evaluation on test datasets <a name="evaluation"></a>

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



## 🏋️ Training

Based on the previously created environment, install extended requirements:

```bash
pip install -r requirements++.txt -r requirements+.txt -r requirements.txt
```

Set environment parameters for the data directory:

```bash
export BASE_DATA_DIR=YOUR_DATA_DIR  # directory of training data
export BASE_CKPT_DIR=YOUR_CHECKPOINT_DIR  # directory of pretrained checkpoint
```

Download Stable Diffusion v2 [checkpoint](https://huggingface.co/sd2-community/stable-diffusion-2) into `${BASE_CKPT_DIR}`

Download the checkpoint of [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) into `DA2/checkpoints/`

Prepare for [Hypersim](https://github.com/apple/ml-hypersim) and [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/) datasets and save into `${BASE_DATA_DIR}`. Please refer to [this README](script/dataset_preprocess/hypersim/README.md) for Hypersim preprocessing.

------------

**Run training script**

```bash
python train.py --config config/train_apdepth.yaml --no_wandb
```

Resume from a checkpoint, e.g.

```bash
python train.py --resume_run output/train_apdepth/checkpoint/latest --no_wandb
```

------------

**Evaluating results**

Only the U-Net is updated and saved during training. To use the inference pipeline with your training result, replace `unet` folder in `train_apdepth` checkpoints with that in the `checkpoint` output folder. Then refer to [this section](#evaluation) for evaluation.

> [!IMPORTANT]
>
> Although random seeds have been set, the training result might be slightly different on different hardwares. It's recommended to train without interruption.



## ✏️ Contributing

Please refer to [this](CONTRIBUTING.md) instruction.

## 🤔 Troubleshooting

| Problem                                                                                                                     | Solution                                                                  |
| --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| (Windows) Invalid DOS bash script on WSL / `$'\r': command not found` / `set: invalid option`                               | Run `dos2unix <script_name>` to convert script format                     |
| (Windows) Multiple `.sh` scripts fail due to CRLF line endings                                                              | Run `find . -name "*.sh" -exec dos2unix {} +` to fix all scripts          |
| (Windows) error on WSL: `Could not load library libcudnn_cnn_infer.so.8. Error: libcuda.so: cannot open shared object file` | Run `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH`            |
| HuggingFace model download incomplete / corrupted                                                                           | Re-run with `--resume-download` or ensure stable network                  |
| `model_index.json not found` when loading checkpoint                                                                        | Ensure the model is fully downloaded and placed at `checkpoints/ApDepth/` |
| Dataset loading error: `tarfile.ReadError: unexpected end of data`                                                          | Re-download dataset; the `.tar` file is likely corrupted or incomplete    |



## 🎓 Citation
Please cite our paper:

```bibtex
@InProceedings{haruko26apdepth,
      title={ApDepth: Aiming for Precise Monocular Depth Estimation Based on Diffusion Models},
      author={Haruko386 and Yuan Shuai},
      booktitle = {Under review},
      year={2026}
}
```

## 🎫 License

This work is licensed under the Apache License, Version 2.0 (as defined in the [LICENSE](LICENSE.txt)).

By downloading and using the code and model you agree to the terms in the  [LICENSE](LICENSE.txt).

[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)
