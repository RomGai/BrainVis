# BrainVis: Exploring the Bridge between Brain and Visual Signals via Image Reconstruction [Link to paper](https://arxiv.org/abs/2312.14871)

![Framework](/figures/framework.jpg)

Abstract: *Analyzing and reconstructing visual stimuli from brain signals effectively advances understanding of the human visual system. However, the EEG signals are complex and contain a amount of noise. This leads to substantial limitations in existing works of visual stimuli reconstruction from EEG, such as difficulties in aligning EEG embeddings with the fine-grained semantic information and a heavy reliance on additional large self-collected dataset for training. To address these challenges, we propose a novel approach called BrainVis. Firstly, we divide the EEG signals into various units and apply a self-supervised approach on them to obtain EEG time-domain features, in an attempt to ease the training difficulty. Additionally, we also propose to utilize the frequency-domain features to enhance the EEG representations. Then, we simultaneously align EEG time-frequency embeddings with the interpolation of the coarse and fine-grained semantics in the CLIP space, to highlight the primary visual components and reduce the cross-modal alignment difficulty. Finally, we adopt the cascaded diffusion models to reconstruct images. Our proposed BrainVis outperforms state of the arts in both semantic fidelity reconstruction and generation quality. Notably, we reduce the training data scale to 10% of the previous work.*

![Results](/figures/results.jpg)

We provide more results [here](https://drive.google.com/file/d/17JFYU-hM1TR1G2ZzR2uZCPx-ChxkDc3z/view?usp=drive_link).

# Preparation

**Environment**

We recommend installing 64-bit Python 3.8 and [PyTorch 1.12.0](https://pytorch.org/get-started/locally/). On a CUDA GPU machine, the following will do the trick:

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
2. Pre-trained stable diffusion model [v1-5-pruned-emaonly](https://huggingface.co/runwayml/stable-diffusion-v1-5). Place the **"v1-5-pruned-emaonly.ckpt"** to path **"/pretrained_model"**.
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

# Broader Information

BrainVis builds upon several previous works:

1. [High-resolution image synthesis with latent diffusion models (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)
2. [Learning Transferable Visual Models From Natural Language Supervision (ICML 2021)](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf)
3. [Seeing beyond the brain: Masked modeling conditioned diffusion model for human vision decoding (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Seeing_Beyond_the_Brain_Conditional_Diffusion_Model_With_Sparse_Masked_CVPR_2023_paper.pdf)
4. [Deep learning human mind for automated visual classification (CVPR 2017)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Spampinato_Deep_Learning_Human_CVPR_2017_paper.pdf)
5. [TimeMAE: Self-Supervised Representations of Time Series with Decoupled Masked Autoencoders](https://arxiv.org/pdf/2303.00320.pdf)

# Citation

```
@article{fu2023brainvis,
    title={BrianVis: Exploring the Bridge between Brain and Visual Signals via Image Reconstruction},
    author={Honghao Fu and Zhiqi Shen and Jing Jih Chin and Hao Wang},
    journal={arXiv preprint arXiv:2312.14871},
    year={2023}
}
```
