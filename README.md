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
   
2. Pretrained stable diffusion model [v1-5-pruned-emaonly](https://huggingface.co/runwayml/stable-diffusion-v1-5). Place the "v1-5-pruned-emaonly.ckpt" in the path "/pretrained_model".
   
3. [EEG-Image pairs dataset](https://tinyurl.com/eeg-visual-classification). Place "block_splits_by_image_all.pth", "block_splits_by_image_single.pth" and "eeg_5_95_std.pth" in the path "/data/EEG".
   
4. A copy of required ImageNet subset. Unzip it in the path "/data/image".

**The training data required for the alignment process**

```
python imageBLIPtoCLIP.py
python imageLabeltoCLIP.py
```

# Train the model
