import numpy as np
import torch
import os
from scipy.io import arff
# Define options
import argparse
from scipy import signal

import pandas as pd

parser = argparse.ArgumentParser(description="Template")
### BLOCK DESIGN ###
#Splits
#parser.add_argument('-sp', '--splits-path', default=r"data\EEG\block_splits_by_image_all.pth", help="splits path") #subjects('all' ←---→ 'single')
parser.add_argument('-sp', '--splits-path', default=r"data\EEG\block_splits_by_image_single.pth", help="splits path") #subjects('all' ←---→ 'single')
### BLOCK DESIGN ###
parser.add_argument('-sn', '--split-num', default=0, type=int, help="split number") #leave this always to zero.
#Subject selecting
parser.add_argument('-sub','--subject', default=1 , type=int, help="choose a subject from 1 to 6, default is 0 (all subjects)")
#Time options: select from 20 to 460 samples from EEG data
parser.add_argument('-tl', '--time_low', default=20, type=float, help="lowest time value")
parser.add_argument('-th', '--time_high', default=460,  type=float, help="highest time value")
# Model type/options
parser.add_argument('-mt','--model_type', default='lstm', help='specify which generator should be used: lstm|EEGChannelNet')
# Parse arguments
opt = parser.parse_args()


def normalize_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)

    return normalized_arr

# Dataset class
class EEGDataset:
    # Constructor
    def __init__(self, eeg_signals_path):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        if opt.subject != 0:
            self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if
                         loaded['dataset'][i]['subject'] == opt.subject]
        else:
            self.data = loaded['dataset']
        self.labels = loaded["labels"]
        self.images = loaded["images"]


        # Compute size
        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[opt.time_low:opt.time_high, :]

        if opt.model_type == "model10":
            eeg = eeg.t()
            eeg = eeg.view(1, 128, opt.time_high - opt.time_low)

        # Get label
        label = self.data[i]["label"]
        #print(label)
        img_name=self.images[self.data[i]["image"]]
        #print(self.images[self.data[i]["image"]])
        # Return
        return eeg, label, img_name

# Splitter class
class Splitter:
    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [i for i in self.split_idx if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        # Compute size
        self.size = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        eeg, label,img_name = self.dataset[self.split_idx[i]]
        # Return
        return eeg, label,img_name

def load_EEG(Path='data/EEG/'):
    clip_path='data/imagelabel_text_CLIP/'
    clip_moreinf_path='data/image_text_CLIP/'
    dataset = EEGDataset(Path+'eeg_5_95_std.pth')
    # # Create loaders
    train_dataset = Splitter(dataset, split_path=opt.splits_path, split_num=opt.split_num, split_name="train")
    val_dataset = Splitter(dataset, split_path=opt.splits_path, split_num=opt.split_num, split_name="val")
    test_dataset = Splitter(dataset, split_path=opt.splits_path, split_num=opt.split_num, split_name="test")

    train_eeg_list = []
    train_label_list = []
    train_img_name_list = []
    train_clip_list=[]
    train_clip_moreinf_list=[]

    for i in range(0, len(train_dataset)):
        eeg, label,img_name = train_dataset[i]
        temp_label = torch.tensor(label)
        train_label_list.append(temp_label)
        train_eeg_list.append(eeg)
        train_img_name_list.append(img_name)

        ###Cancel the annotation of this section during CLIP alignment
        clip_file = clip_path + img_name + '.csv'
        target_vec = torch.tensor(pd.read_csv(clip_file, header=None).values).reshape(1, -1).squeeze()

        ### Please annotate this during CLIP alignment
        #target_vec=torch.tensor([0,0,0,0,0])

        train_clip_list.append(target_vec)

        ###Cancel the annotation of this section during CLIP alignment
        clip_moreinf_file = clip_moreinf_path + img_name + '.csv'
        target_moreinf_vec = torch.tensor(pd.read_csv(clip_moreinf_file, header=None).values).reshape(1, -1).squeeze()

        ### Please annotate this during CLIP alignment
        #target_moreinf_vec=torch.tensor([0,0,0,0,0])

        train_clip_moreinf_list.append(target_moreinf_vec)

    TRAIN_DATA = torch.stack(train_eeg_list, dim=0)
    TRAIN_LABEL = torch.stack(train_label_list, dim=0)
    TRAIN_IMG_NAME=train_img_name_list
    TRAIN_CLIP=torch.stack(train_clip_list, dim=0)
    TRAIN_MOREINF_CLIP= torch.stack(train_clip_moreinf_list, dim=0)

    val_eeg_list = []
    val_label_list = []
    val_img_name_list=[]
    val_clip_list=[]
    val_clip_moreinf_list=[]

    for i in range(0, len(val_dataset)):
        eeg, label,img_name = val_dataset[i]
        temp_label = torch.tensor(label)
        val_label_list.append(temp_label)
        val_eeg_list.append(eeg)
        val_img_name_list.append(img_name)
        clip_file = clip_path + img_name + '.csv'
        target_vec = torch.tensor(pd.read_csv(clip_file, header=None).values).reshape(1, -1).squeeze()
        #target_vec = torch.tensor([0, 0, 0, 0, 0])
        val_clip_list.append(target_vec)
        clip_moreinf_file = clip_moreinf_path + img_name + '.csv'
        target_moreinf_vec = torch.tensor(pd.read_csv(clip_moreinf_file, header=None).values).reshape(1, -1).squeeze()
        #target_moreinf_vec=torch.tensor([0,0,0,0,0])
        val_clip_moreinf_list.append(target_moreinf_vec)

    VAL_DATA = torch.stack(val_eeg_list, dim=0)
    VAL_LABEL = torch.stack(val_label_list, dim=0)
    VAL_IMG_NAME=val_img_name_list
    VAL_CLIP = torch.stack(val_clip_list, dim=0)
    VAL_MOREINF_CLIP= torch.stack(val_clip_moreinf_list, dim=0)

    # 创建空列表 for test data
    test_eeg_list = []
    test_label_list = []
    test_img_name_list = []
    test_clip_list = []
    test_clip_moreinf_list = []

    for i in range(0, len(test_dataset)):
        eeg, label,img_name = test_dataset[i]
        temp_label = torch.tensor(label)
        test_label_list.append(temp_label)
        test_eeg_list.append(eeg)
        test_img_name_list.append(img_name)
        clip_file = clip_path + img_name + '.csv'
        target_vec = torch.tensor(pd.read_csv(clip_file, header=None).values).reshape(1, -1).squeeze()
        #target_vec = torch.tensor([0, 0, 0, 0, 0])
        test_clip_list.append(target_vec)
        clip_moreinf_file = clip_moreinf_path + img_name + '.csv'
        target_moreinf_vec = torch.tensor(pd.read_csv(clip_moreinf_file, header=None).values).reshape(1, -1).squeeze()
        #target_moreinf_vec=torch.tensor([0,0,0,0,0])
        test_clip_moreinf_list.append(target_moreinf_vec)

    TEST_DATA = torch.stack(test_eeg_list, dim=0)
    TEST_LABEL = torch.stack(test_label_list, dim=0)
    TEST_CLIP = torch.stack(test_clip_list, dim=0)
    TEST_MOREINF_CLIP= torch.stack(test_clip_moreinf_list, dim=0)

    ALL_TRAIN_DATA = torch.cat([TRAIN_DATA, VAL_DATA])
    ALL_TRAIN_LABEL = torch.cat([TRAIN_LABEL, VAL_LABEL])
    ALL_TRAIN_CLIP = torch.cat([TRAIN_CLIP, VAL_CLIP])
    ALL_TRAIN_MOREINF_CLIP=torch.cat([TRAIN_MOREINF_CLIP, VAL_MOREINF_CLIP])
    print('data loaded')

    return [np.array(ALL_TRAIN_DATA), np.array(ALL_TRAIN_LABEL),np.array(ALL_TRAIN_CLIP),np.array(ALL_TRAIN_MOREINF_CLIP)], \
           [np.array(TRAIN_DATA), np.array(TRAIN_LABEL),np.array(TRAIN_CLIP),np.array(TRAIN_MOREINF_CLIP)], \
           [np.array(TEST_DATA), np.array(TEST_LABEL),np.array(TEST_CLIP),np.array(TEST_MOREINF_CLIP)]

def load_EEG_with_img_name(Path='data/EEG/'):
    clip_path = 'data/imagelabel_text_CLIP/'
    dataset = EEGDataset(Path + 'eeg_5_95_std.pth')
    # # Create loaders
    train_dataset = Splitter(dataset, split_path=opt.splits_path, split_num=opt.split_num, split_name="train")
    val_dataset = Splitter(dataset, split_path=opt.splits_path, split_num=opt.split_num, split_name="val")
    test_dataset = Splitter(dataset, split_path=opt.splits_path, split_num=opt.split_num, split_name="test")

    train_eeg_list = []
    train_label_list = []
    train_img_name_list = []

    for i in range(0, len(train_dataset)):
        eeg, label, img_name = train_dataset[i]
        temp_label = torch.tensor(label)
        train_label_list.append(temp_label)
        train_eeg_list.append(eeg)
        train_img_name_list.append(img_name)

    TRAIN_DATA = torch.stack(train_eeg_list, dim=0)
    TRAIN_LABEL = torch.stack(train_label_list, dim=0)
    TRAIN_IMG_NAME = train_img_name_list

    val_eeg_list = []
    val_label_list = []
    val_img_name_list = []

    for i in range(0, len(val_dataset)):
        eeg, label, img_name = val_dataset[i]
        temp_label = torch.tensor(label)
        val_label_list.append(temp_label)
        val_eeg_list.append(eeg)
        val_img_name_list.append(img_name)

    VAL_DATA = torch.stack(val_eeg_list, dim=0)
    VAL_LABEL = torch.stack(val_label_list, dim=0)
    VAL_IMG_NAME = val_img_name_list

    test_eeg_list = []
    test_label_list = []
    test_img_name_list = []

    for i in range(0, len(test_dataset)):
        eeg, label, img_name = test_dataset[i]
        temp_label = torch.tensor(label)
        test_label_list.append(temp_label)
        test_eeg_list.append(eeg)
        test_img_name_list.append(img_name)

    TEST_DATA = torch.stack(test_eeg_list, dim=0)
    TEST_LABEL = torch.stack(test_label_list, dim=0)
    TEST_IMG_NAME = test_img_name_list

    ALL_TRAIN_DATA = torch.cat([TRAIN_DATA, VAL_DATA])
    ALL_TRAIN_LABEL = torch.cat([TRAIN_LABEL, VAL_LABEL])
    ALL_TRAIN_IMG_NAME = TRAIN_IMG_NAME + VAL_IMG_NAME
    print('data loaded')


    return [np.array(ALL_TRAIN_DATA), np.array(ALL_TRAIN_LABEL),ALL_TRAIN_IMG_NAME], [np.array(TRAIN_DATA), np.array(TRAIN_LABEL),TRAIN_IMG_NAME], [
        np.array(TEST_DATA), np.array(TEST_LABEL),TEST_IMG_NAME]




