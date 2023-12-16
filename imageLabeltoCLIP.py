import torch
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

propmt_dict = {'n02106662': 'german shepherd dog',
'n02124075': 'cat ',
'n02281787': 'lycaenid butterfly',
'n02389026': 'sorrel horse',
'n02492035': 'Cebus capucinus',
'n02504458': 'African elephant',
'n02510455': 'panda',
'n02607072': 'anemone fish',
'n02690373': 'airliner',
'n02906734': 'broom',
'n02951358': 'canoe or kayak',
'n02992529': 'cellular telephone',
'n03063599': 'coffee mug',
'n03100240': 'old convertible',
'n03180011': 'desktop computer',
'n03197337': 'digital watch',
'n03272010': 'lectric guitar',
'n03272562': 'electric locomotive',
'n03297495': 'espresso maker',
'n03376595': 'folding chair',
'n03445777': 'golf ball',
'n03452741': 'grand piano',
'n03584829': 'smoothing iron',
'n03590841': 'orange jack-o’-lantern',
'n03709823': 'mailbag',
'n03773504': 'missile',
'n03775071': 'mitten,glove',
'n03792782': 'mountain bike, all-terrain bike',
'n03792972': 'mountain tent',
'n03877472': 'pajama',
'n03888257': 'parachute',
'n03982430': 'pool table, billiard table, snooker table ',
'n04044716': 'radio telescope',
'n04069434': 'eflex camera',
'n04086273': 'revolver, six-shooter',
'n04120489': 'running shoe',
'n07753592': 'banana',
'n07873807': 'pizza',
'n11939491': 'daisy',
'n13054560': 'bolete'
}

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

    generated_text = propmt_dict[folder]

    with torch.no_grad():

        batch_encoding = tokenizer(generated_text, truncation=True, max_length=max_length, return_length=True,
                               return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"]  # .to(self.device)
        outputs = transformer(input_ids=tokens)

        z = outputs.last_hidden_state.squeeze(0)

    z=z.cpu()
    np.savetxt('data/imagelabel_text_CLIP/'+imgname+'.csv', z, delimiter=',')
