import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torchvision.models import ViT_H_14_Weights, vit_h_14
import numpy as np
from clip import clip
from torchmetrics.functional import accuracy
import os
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import cosine
from torchvision.models import inception_v3
from torchvision.transforms import ToTensor, Normalize
from skimage import io, color
from torch.utils.data import DataLoader
import torch.nn as nn

def preprocess_images(images):
    transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor_images = torch.stack([ToTensor()(img) for img in images])
    normalized_images = transform(tensor_images)
    return normalized_images

def calculate_inception_score(images, batch_size=32):
    model = inception_v3(pretrained=True, transform_input=False).to("cuda")
    model.eval()

    normalized_images = preprocess_images(images)

    dataset = torch.utils.data.TensorDataset(normalized_images)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    predictions = []
    pos=0
    with torch.no_grad():
        for batch in dataloader:
            pos=pos+1
            inputs = torch.tensor(batch[0]).to("cuda")
            pred=model(inputs)
            preds=nn.functional.softmax(pred, dim=1)
            predictions.append(preds)
            print(str(pos)+"/"+str(len(dataloader)))

    scores = torch.cat(predictions, dim=0)
    avg_scores = scores.mean(dim=0)
    kl_scores = scores * (torch.log(scores+1E-16) - torch.log(avg_scores+1E-16))
    kl_divergence = kl_scores.sum(dim=1)
    realness = kl_divergence.mean().exp()

    diversity = kl_divergence.exp().mean()

    inception_score = realness * diversity

    return inception_score.item()

IS_list=[]
gt_list=[]
generated_image_list=[]
gt_folder = "picture-gene-onlygt"
gt_images_name = os.listdir(gt_folder)
gt_images_name.sort()

k=0
num_samples=4

for gt_name in gt_images_name:

    k=k+1
    print(str(k)+"/"+str(len(gt_images_name)))

    real_image = Image.open('picture-gene-onlygt/' + gt_name).convert('RGB')

    gene_image_name=[]
    name1 = gt_name.split('_')[0] + '_' + gt_name.split('_')[1] + '_' + gt_name.split('_')[2] + '_' + \
            gt_name.split('_')[3] + '_' + gt_name.split('_')[4] + '_' + gt_name.split('_')[5] + '_1.png'
    name2 = gt_name.split('_')[0] + '_' + gt_name.split('_')[1] + '_' + gt_name.split('_')[2] + '_' + \
            gt_name.split('_')[3] + '_' + gt_name.split('_')[4] + '_' + gt_name.split('_')[5] + '_2.png'
    name3 = gt_name.split('_')[0] + '_' + gt_name.split('_')[1] + '_' + gt_name.split('_')[2] + '_' + \
            gt_name.split('_')[3] + '_' + gt_name.split('_')[4] + '_' + gt_name.split('_')[5] + '_3.png'
    name4 = gt_name.split('_')[0] + '_' + gt_name.split('_')[1] + '_' + gt_name.split('_')[2] + '_' + \
            gt_name.split('_')[3] + '_' + gt_name.split('_')[4] + '_' + gt_name.split('_')[5] + '_4.png'
    gene_image_name.append(name1)
    gene_image_name.append(name2)
    gene_image_name.append(name3)
    gene_image_name.append(name4)

    gt = real_image

    for i in range(0,num_samples):
        try:
            with Image.open('subj6pic/stage2/' + gene_image_name[i]).convert('RGB') as generat_image:
                generated_image = generat_image
        except:
            print(f"Failed to read image: {gene_image_name[i]}")
            continue
        generated_image_list.append(generated_image)

IS=calculate_inception_score(generated_image_list,32)

print(IS)




