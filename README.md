## Introduction

**Disclaimer**: This repository is forked from Facebook's [MMF](https://github.com/facebookresearch/mmf) and has been modified to obtain certain interpretabilty measures from state-of-the-art multimodal models (namely, VisualBERT and ViLBERT) for hateful memes detection. If you intend to work on Facebook's hateful memes detection tasks, do refer to the following materials.
- [The hateful memes challenge: detecting hate speech in multimodal memes](https://dl.acm.org/doi/abs/10.5555/3495724.3495944)
- [Facebook's MMF repository on Hateful Memes Detection](https://github.com/facebookresearch/mmf/tree/main/projects/hateful_memes)

**Purpose**: This repository incorporates Facebook's open-sourced [Captum](https://captum.ai) library for interpretability and serves as a fundamental step for paper reproducibility

## Repository Usage
1. [Feature Extraction](#feature-extraction)
2. [Attentions Generation](#attentions-generation)

### Feature Extraction (i.e. converting lmdb to features.npy)

Facebook's MMF provides a Lightning Memory-Mapped Database Manager (lmdb) that contains the visual features for the Facebook's Hateful Memes dataset. To convert the lmdb file to easily readable numpy files, you can use the provided tool using the following command:

```python
python3 tools/scripts/features/lmdb_conversion.py \
    --mode extract \
    --lmdb_path {path_to_lmdb_file} \
    --features_folder {path_to_output_folder}
```

### Attentions Generation

#### Step 1: Prerequisites

Follow the prerequisites instructions from the original repository: https://github.com/facebookresearch/mmf/tree/main/projects/hateful_memes. This will provide you the datasets from "The Hateful Memes Challenge", which is required for the pretrained model inference.

**Note: You do not need to install the MMF from the original repository, as we will be installing this repository instead**

#### Step 2. Install the Pytorch requirements

In our experiments, we used the following Pytorch environments.

```
- torch==1.9.1+cu111 
- torchvision==0.10.1+cu111 
- torchaudio==0.9.1
```

While we expect this repository to work with newer pytorch versions, it is recommended that you install our pytorch environments to avoid possible compatibility problems.

#### Step 3. Install from source

**Important Note:** You need to uninstall existing *mmf* package prior to installing this repostiory. The installation docuemntation is largely adopted from the original MMF's documentation.

```bash
  git clone https://github.com/mingshanhee/ExplainHatefulMeme-mmf.git
  cd mmf
  pip install --editable .
```

In case you met permission issues when running the command, try to install to user folder and disable build isolation.

```
pip install --editable . --user --no-build-isolation
```

### Step 3. Modify and run the prepared scripts

To facilitate easier adoption by fellow researchers, we have prepared and released sample script on how to perform inference using Facebook's Pretrained Models on Hateful Memes

```bash
# VisualBERT
bash sample_scripts/test_visualbert_pretrained.sh

# ViLBERT
bash sample_scripts/test_vilbert_pretrained.sh
```