import time
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import os
import numpy as np
from clip import clip
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

modelclip, preprocess = clip.load('ViT-L/14', device=device)
modelclip.eval()

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
)
model.to(device)

data = torch.load('data/EEG/eeg_5_95_std.pth')

dataset=data.get('dataset')
labels=data.get('labels')
images=data.get('images')

max_length = 77

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
transformer = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

#model.to(device)
for i in range(0,len(dataset)):
    print(str(i+1)+'/'+str(len(dataset)))
    root='data/image/'
    folder=labels[dataset[i].get('label')]
    imgname=images[dataset[i].get('image')]
    type='.JPEG'
    image_path = os.path.join(root,folder, imgname)
    image_path=image_path+type
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(generated_text)
    with open('data/image_text/'+imgname+'.txt', 'w') as file:
        file.write(generated_text)

    with torch.no_grad():

        batch_encoding = tokenizer(generated_text, truncation=True, max_length=max_length, return_length=True,
                               return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"]  # .to(self.device)
        outputs = transformer(input_ids=tokens)

        z = outputs.last_hidden_state.squeeze(0)

    print(z)
    print(np.shape(z))
    z=z.cpu()
    np.savetxt('data/image_text_CLIP/'+imgname+'.csv', z, delimiter=',')
