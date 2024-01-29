import torch
import os
from omegaconf import OmegaConf
from dc_ldm.models.diffusion.plms import PLMSSampler
from einops import rearrange, repeat
import numpy as np
from dc_ldm.util import instantiate_from_config
from torch.utils.data import Dataset, DataLoader
from dataset import Dataset as selfdataset
import torchvision.transforms as transforms
from model.BrainVisModels import TimeEncoder, AlignNet,TimeFreqEncoder,FreqEncoder
from args import args, Test_data, Train_data_all, Train_data, Train_data_all_with_image_name, Train_data_with_image_name, Test_data_with_image_name
import argparse
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

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
'n03272010': 'electric guitar',
'n03272562': 'electric locomotive',
'n03297495': 'espresso maker',
'n03376595': 'folding chair',
'n03445777': 'golf ball',
'n03452741': 'grand piano',
'n03584829': 'smoothing iron',
'n03590841': 'Orange jack-o’-lantern',
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

lable_number_dict={
'[12]': 'n02106662',
'[39]': 'n02124075',
'[11]': 'n02281787',
'[0]': 'n02389026',
'[21]': 'n02492035',
'[35]': 'n02504458',
'[8]': 'n02510455',
'[3]': 'n02607072',
'[36]': 'n02690373',
'[18]': 'n02906734',
'[10]': 'n02951358',
'[15]': 'n02992529',
'[5]': 'n03063599',
'[24]': 'n03100240',
'[17]': 'n03180011',
'[34]': 'n03197337',
'[28]': 'n03272010',
'[37]': 'n03272562',
'[4]': 'n03297495',
'[25]': 'n03376595',
'[16]': 'n03445777',
'[30]': 'n03452741',
'[2]': 'n03584829',
'[14]': 'n03590841',
'[23]': 'n03709823',
'[20]': 'n03773504',
'[27]': 'n03775071',
'[6]': 'n03792782',
'[31]': 'n03792972',
'[26]': 'n03877472',
'[1]': 'n03888257',
'[22]': 'n03982430',
'[38]': 'n04044716',
'[29]': 'n04069434',
'[7]': 'n04086273',
'[13]': 'n04120489',
'[32]': 'n07753592',
'[19]': 'n07873807',
'[9]': 'n11939491',
'[33]': 'n13054560'
}

parser = argparse.ArgumentParser(description="Template")

parser.add_argument('-mp','--model_params', default='', nargs='*', help='list of key=value pairs of model options')
opt = parser.parse_args()

# Path
datapath='data/EEG_Feature_Label/'
img_file_type='.JPEG'
device = "cuda"

test_img_names_file=datapath+'test_image_names.pth'
test_seq_file=datapath+'test_seqs.pth'
dff_model_path = "pretrained_model/v1-5-pruned-emaonly.ckpt"
dff_yaml_path = "pretrained_model/config15.yaml"
test_pred_file=datapath+'test_pred.pth'
output_path="picture"

logger=None
ddim_steps=40
global_pool=True
use_time_cond=False
clip_tune=False
cls_tune=False
ddim_steps=50

def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img

def channel_last(img):
    if img.shape[-1] == 3:
        return img
    return rearrange(img, 'c h w -> h w c')

class Dataset(Dataset):
    def __init__(self, img_names_file,seq_file,labels_file):
        self.image_names = torch.load(img_names_file)
        self.seqs = torch.load(seq_file)
        self.labels = torch.load(labels_file)

    def __len__(self):
            return len(self.seqs)

    def __getitem__(self, idx):
        input_vec=torch.tensor(self.seqs[idx]).to("cuda")

        img_label = self.image_names[idx].split("_")[0]
        img_path = "data/image/" + img_label + "/" + self.image_names[idx] + img_file_type
        image = Image.open(img_path).convert('RGB')

        img_transform_test = transforms.Compose([
            normalize,
            transforms.Resize((512, 512)),
            channel_last
        ])
        gt_image = np.array(image) / 255.0
        gt_image = img_transform_test(gt_image)

        prompt = propmt_dict[lable_number_dict[str(self.labels[idx])]]

        return input_vec, gt_image,self.image_names[idx],prompt



#Load data
batch_size = 1
test_dataset = Dataset(test_img_names_file,test_seq_file, test_pred_file)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

train_dataset = selfdataset(device=args.device, mode='pretrain', data=Train_data_all, wave_len=args.wave_length)
args.data_shape = train_dataset.shape()

#Load AlignNet
time_model=TimeEncoder(args)
time_model=time_model.to("cuda")
freq_model_options = {key: int(value) if value.isdigit() else (float(value) if value[0].isdigit() else value) for
                      (key, value) in [x.split("=") for x in opt.model_params]}

freq_model = FreqEncoder(**freq_model_options)

timefreq_model = TimeFreqEncoder(time_model, freq_model, args)
timefreq_model=timefreq_model.to("cuda")

time_size=1024
freq_size=128
clip_size=int(77*768)

model_eegtoclip=AlignNet(time_size,freq_size,clip_size,timefreq_model)
eegtoclip_state_dict = torch.load('exp/epilepsy/test/clipfinetune_model.pkl', map_location="cuda")#device)
model_eegtoclip.load_state_dict(eegtoclip_state_dict)
model_eegtoclip.to("cuda")
model_eegtoclip.eval()

#Load stable diffusion
ckp_path = os.path.join(dff_model_path)
config_path = os.path.join(dff_yaml_path)
config = OmegaConf.load(config_path)
config.model.params.unet_config.params.use_time_cond = use_time_cond
config.model.params.unet_config.params.global_pool = global_pool
cond_dim = config.model.params.unet_config.params.context_dim
model = instantiate_from_config(config.model)
pl_sd = torch.load(ckp_path, map_location=device)['state_dict']
m, u = model.load_state_dict(pl_sd, strict=False)
model.cond_stage_trainable = False
model.ddim_steps = ddim_steps
model.re_init_ema()
model.p_channels = config.model.params.channels
model.p_image_size = config.model.params.image_size
model.ch_mult = config.model.params.first_stage_config.params.ddconfig.ch_mult
model.clip_tune = clip_tune
model.cls_tune = cls_tune
model = model.to(device)
sampler = PLMSSampler(model)
ldm_config = config
shape = (ldm_config.model.params.channels, ldm_config.model.params.image_size, ldm_config.model.params.image_size)

#Verify classification results
labels = torch.load(test_pred_file)
image_names = torch.load(test_img_names_file)

errnum=0

for idx in range(0,len(labels)):
    nowclass = image_names[idx].split('_')[0]
    nowlabel = labels[idx]
    now_label_name = lable_number_dict[str(nowlabel)]
    if (now_label_name != nowclass):
        errnum = errnum + 1
        print(idx)

print("errclassnum:"+str(errnum))

model2 = StableDiffusionImg2ImgPipeline.from_single_file("pretrained_model/v1-5-pruned-emaonly.ckpt").to(device)


num_samples=4
gt_img_num=1
agt_img_for_num=4

all_samples = []

curr_epoch=0

batch_pos=0
now_batch=0

print(len(test_loader))

for inputs, gt_image, img_name, prompt in test_loader:
    output_path = "picture"
    gt_img = []
    latent = []
    images_list = []
    curr_epoch = curr_epoch + 1
    inputs = inputs.float()
    clip_encoded = model_eegtoclip(inputs)

    with model.ema_scope():
        model.eval()
        for index in range(0, len(inputs)):
            cur_gt_img = rearrange(gt_image[index], 'h w c -> c h w')
            gt_img.append(torch.clamp((cur_gt_img + 1.0) / 2.0, min=0.0, max=1.0))
            latent.append(clip_encoded[index].reshape(77, 768).to(device))

        index = 0

        for i in range(0, len(inputs)):
            prompt_list = []
            single_c = latent[i].unsqueeze(dim=0)
            c = torch.cat((single_c, single_c, single_c, single_c), dim=0)
            print(np.shape(c))
            print(f"rendering {num_samples} examples in {ddim_steps} steps.")

            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=c,
                                             batch_size=num_samples,
                                             shape=shape,
                                             verbose=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples_to_stage2=255.*x_samples_ddim

            samples_stage2 = (255. * x_samples_ddim.detach().cpu().numpy()).astype(np.uint8)

            if output_path is not None:
                samples_t = (
                        255. * torch.cat([gt_img[i].unsqueeze(dim=0).detach().cpu(), x_samples_ddim.detach().cpu()],
                                         dim=0).numpy()).astype(np.uint8)
                for copy_idx, img_t in enumerate(samples_t):
                    img_t = rearrange(img_t, 'c h w -> h w c')
                    if (copy_idx == 0):
                        Image.fromarray(img_t).save(
                            "picture-gene/batch_" + str(batch_pos) + "_test_" + str(index) + "_" + img_name[
                                index] + "_gt.png")
                        Image.fromarray(img_t).save(
                            "picture-gene-onlygt/batch_" + str(batch_pos) + "_test_" + str(index) + "_" + img_name[
                                index] + "_gt.png")

            for k in range(0,num_samples):
                prompt_list.append(prompt[index])

            print("stage2")

            for copy_idx, img_t in enumerate(samples_stage2):
                img_t = rearrange(img_t, 'c h w -> h w c')
                images_list.append(img_t)

            generated_image = model2(prompt_list, images_list, strength=0.75, guidance_scale=7.5).images#, num_inference_steps=100).images

            for j in range(0, num_samples):
                generated_image[j].save("picture-gene/batch_" + str(batch_pos) + "_test_" + str(index) + "_" + img_name[
                        index] + "_" + str(j+1) + ".png")

            index = index + 1

        batch_pos=batch_pos+1

