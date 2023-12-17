# BrainVis: Exploring the Bridge between Brain and Visual Signals via Image Reconstruction

-----Abstract and the figure of the pipeline-----

-----Reconstruction results display-----

# Preparation

**Environment**

We recommend installing 64-bit Python 3.8 and PyTorch 1.12.0. See https://pytorch.org for PyTorch install instructions. On a CUDA GPU machine, the following will do the trick:

```
pip install ftfy
pip install omegaconf
pip install einops
pip install torchmetrics
pip install pytorch-lightning
pip install transformers
pip install kornia
pip install diffusers
```

**Create paths**

```
python create_path.py
```

**Download required files**

1. [CLIP](https://github.com/openai/CLIP). Place the "clip" folder in this project.
2. Pre-trained stable diffusion model [v1-5-pruned-emaonly](https://huggingface.co/runwayml/stable-diffusion-v1-5). Place the "v1-5-pruned-emaonly.ckpt" in the path "/pretrained_model".
3. [EEG-Image pairs dataset](https://tinyurl.com/eeg-visual-classification). Place "block_splits_by_image_all.pth", "block_splits_by_image_single.pth" and "eeg_5_95_std.pth" in the path "/data/EEG".
4. [Copy](https://drive.google.com/file/d/1k3Psdqhl0Saiol4Yauy6eCQK6_-Em05R/view?usp=drive_link) of required ImageNet subset. Unzip it in the path "/data/image".

**Obtain the training data required for the alignment process**

```
python imageBLIPtoCLIP.py
python imageLabeltoCLIP.py
```

# Train the model

1. Run `train_freqencoder.py` to train the frequency encoder.
2. Run `main.py` to pre-train the time encoder.
3. Comment out **"trainer.pretrain()"** on **line 59** of `main.py`, and uncomment **"trainer.finetune()"** on **line 61**. Run `main.py` to fine-tune the time encoder.
4. Modify **"_all"** to **"_single"** in **line 14** of `datautils.py`, and change **"default=0"** to any number from 1 to 6 in **line 19** to use a different single subject. Comment out **line 61** in `main.py` and uncomment **"trainer.finetune_timefreq()"** on line 64. Run `main.py` to integrate the time and frequency models.
5. Comment out **line 64** of `main.py`, and uncomment **"trainer.finetune_CLIP()"** on **line 65**. Run `main.py` to conduct cross-modal EEG alignment.
6. Modify the **"train_mode"** to **"False"** on **line 56** of `main.py` and run it to save the alignment results for reconstruction.

# Image Reconstruction

```
python cascade_diffusion.py
```

Results will be saved in the path "/picture-gene".

# Acknowledgement

We thank these authors of [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [EEG Visual Classification](https://github.com/perceivelab/eeg_visual_classification), [TimeMAE](https://github.com/Mingyue-Cheng/TimeMAE), [Mind-Vis](https://github.com/zjc062/mind-vis) and [CLIP](https://github.com/openai/CLIP) for making thier code publicly available. and the [PeRCeiVe Lab](https://www.perceivelab.com/) for making their raw and pre-processed data public.

# Citation

```
-----cite-----
```

