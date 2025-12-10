# [ICASSP 2025] BrainVis: Exploring the Bridge between Brain and Visual Signals via Image Reconstruction [[Link to paper](https://arxiv.org/abs/2312.14871)]

![Framework](/figures/framework.jpg)

Abstract: *Analyzing and reconstructing visual stimuli from brain signals effectively advances our understanding of the human visual system. However, EEG signals are complex and contain significant noise, leading to substantial limitations in existing approaches of visual stimuli reconstruction from EEG. These limitations include difficulties in aligning EEG embeddings with fine-grained semantic information and a heavy reliance on additional large-scale datasets for training. To address these challenges, we propose a novel approach called BrainVis. This approach introduces a self-supervised paradigm to learn EEG time-domain features and incorporates frequency-domain features to enhance EEG representations. We also propose a multi-modal alignment method called semantic interpolation to achieve fine-grained semantic reconstruction. Additionally, we employ cascaded diffusion models to reconstruct images. Using only 9.1\% of the training data required by previous mask modeling works, our proposed BrainVis outperforms state-of-the-art methods in both semantic fidelity reconstruction and generation quality.*

![Results](/figures/results.jpg)

# Preparation

**Environment**

We recommend installing 64-bit Python 3.10.12 and [PyTorch 2.5.1](https://pytorch.org/get-started/locally/). On a CUDA GPU machine, the following will do the trick:

```
pip install numpy==1.26.0
pip install ftfy==6.2.0
pip install omegaconf==2.3.0
pip install einops==0.8.0
pip install torchmetrics==1.4.0.post0
pip install pytorch-lightning==2.3.3
pip install transformers==4.42.4
pip install kornia==0.7.3
pip install diffusers==0.29.2
```

We have done all testing and development using A100 GPU.

**Create paths**

```
python create_path.py
```

**Download required files**

1. [CLIP](https://github.com/openai/CLIP). Place the **"clip"** folder in this project.
2. Pre-trained stable diffusion model [v1-5-pruned-emaonly](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5). Place the **"v1-5-pruned-emaonly.ckpt"** to path **"/pretrained_model"**.
3. [EEG-Image pairs dataset](https://tinyurl.com/eeg-visual-classification). Place **"block_splits_by_image_all.pth", "block_splits_by_image_single.pth" and "eeg_5_95_std.pth"** to path **"/data/EEG"**.
4. A [copy](https://drive.google.com/file/d/1k3Psdqhl0Saiol4Yauy6eCQK6_-Em05R/view?usp=drive_link) of required ImageNet subset. Unzip it to path **"/data/image"**.

**Obtain the training data required for the alignment process**

```
python imageBLIPtoCLIP.py
python imageLabeltoCLIP.py
```

# Train the model

1. Run `train_freqencoder.py` to train the frequency encoder.
2. Run `main.py` to pre-train the time encoder.
3. Comment out **"trainer.pretrain()"** on **line 59** of `main.py`, and uncomment **"trainer.finetune()"** on **line 61**. Run `main.py` to fine-tune the time encoder.
4. Modify **"_all"** to **"_single"** in **line 14** of `datautils.py`, and change **"default=0"** to any number from 1 to 6 in **line 19** to use a different single subject. Comment out **line 61** in `main.py` and uncomment **"trainer.finetune_timefreq()"** on **line 64**. Run `main.py` to integrate the time and frequency models.
5. Comment out **line 64** of `main.py`, and uncomment **"trainer.finetune_CLIP()"** on **line 65**. Run `main.py` to conduct cross-modal EEG alignment.
6. Modify the **"train_mode="** to **"False"** on **line 56** of `main.py` and run it to save the alignment results for reconstruction.

# Image Reconstruction

```
python cascade_diffusion.py
```

Results will be saved in the path "/picture-gene".

# Checkpoints

Please leave your email address in the issue (we will respond as soon as possible), or contact us directly via email at hfu006@e.ntu.edu.sg (some emails might be missed). We will send you the checkpoint along with the usage instructions.

# Broader Information

BrainVis builds upon several previous works:

1. [High-resolution image synthesis with latent diffusion models (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)
2. [Learning Transferable Visual Models From Natural Language Supervision (ICML 2021)](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf)
3. [Seeing beyond the brain: Masked modeling conditioned diffusion model for human vision decoding (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Seeing_Beyond_the_Brain_Conditional_Diffusion_Model_With_Sparse_Masked_CVPR_2023_paper.pdf)
4. [Deep learning human mind for automated visual classification (CVPR 2017)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Spampinato_Deep_Learning_Human_CVPR_2017_paper.pdf)
5. [TimeMAE: Self-Supervised Representations of Time Series with Decoupled Masked Autoencoders](https://arxiv.org/pdf/2303.00320.pdf)

# Citation

If you find BrainVis useful for your research, we would greatly appreciate it if you could star it on GitHub and cite using this BibTeX.

```
@inproceedings{fu2025brainvis,
  title={BrainVis: Exploring the bridge between brain and visual signals via image reconstruction},
  author={Fu, Honghao and Wang, Hao and Chin, Jing Jih and Shen, Zhiqi},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```
