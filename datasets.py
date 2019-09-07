import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import skimage.io as io
import glob
import numpy as np
import random
import pdb
import sys
# sys.path.append('/home/gongxp/mlmr/githubcode/super-resolution/')
# from SRCNN.model import *
def get_images(filename):
    image_names = np.genfromtxt(filename, dtype=str)
    return image_names


def load_image(file):
    return Image.open(file)


class coseg_train_dataset(Dataset):
    def __init__(self, data_dir, label_dir, traintxt, input_transform=None, label_transform=None):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.input_transform = input_transform
        self.label_transform = label_transform
        self.traintxt = traintxt
        self.train_names = get_images(self.traintxt)

        # self.pre_model = torch.load('/home/gongxp/mlmr/githubcode/super-resolution/SRCNN/model_path/SRCNN_singnal_model_path_1x_epoch1.pth').cuda()
        # self.pre_model.eval()
    def __getitem__(self, index):
        name_array = self.train_names[index].split(',')
        name1 = name_array[0]
        name2 = name_array[1]
        imagename1 = self.data_dir + name1 + ".jpg"
        imagename2 = self.data_dir + name2 + ".jpg"
        labelname1 = self.label_dir + name1 + ".png"
        labelname2 = self.label_dir + name2 + ".png"

        with open(imagename1, "rb") as f:
            image1 = load_image(f).convert('RGB')
        with open(imagename2, "rb") as f:
            image2 = load_image(f).convert('RGB')

        with open(labelname1, "rb") as f:
            label1 = load_image(f).convert('L')
        with open(labelname2, "rb") as f:
            label2 = load_image(f).convert('L')

        # random horizontal flip
        if random.random() < 0.5:
            image1 = image1.transpose(Image.FLIP_LEFT_RIGHT)
            label1 = label1.transpose(Image.FLIP_LEFT_RIGHT)

        # random horizontal flip
        if random.random() < 0.5:
            image2 = image2.transpose(Image.FLIP_LEFT_RIGHT)
            label2 = label2.transpose(Image.FLIP_LEFT_RIGHT)
        if self.input_transform is not None:
            
            image1 = self.input_transform(image1)
            image2 = self.input_transform(image2)

        if self.label_transform is not None:
            label1 = self.label_transform(label1)
            label2 = self.label_transform(label2)
            # label1 = torch.squeeze(torch.squeeze(self.pre_model(torch.unsqueeze(torch.unsqueeze(label1.cuda(), dim=0),dim=0)), dim=0), dim=0)
            # label2 = torch.squeeze(torch.squeeze(self.pre_model(torch.unsqueeze(torch.unsqueeze(label2.cuda(), dim=0)), dim=0), dim=0),dim=0)

        return image1, image2, label1, label2

    def __len__(self):
        return len(self.train_names)


class coseg_val_dataset(Dataset):
    def __init__(self, data_dir, label_dir, val_txt, input_transform=None, label_transform=None):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.input_transform = input_transform
        self.label_transform = label_transform
        self.val_txt = val_txt
        self.val_names = get_images(self.val_txt)

    def __getitem__(self, index):
        name_array = self.val_names[index].split(',')
        name1 = name_array[0]
        name2 = name_array[1]
        imagename1 = self.data_dir + name1 + ".jpg"
        imagename2 = self.data_dir + name2 + ".jpg"
        labelname1 = self.label_dir + name1 + ".png"
        labelname2 = self.label_dir + name2 + ".png"

      

        with open(imagename1, "rb") as f:
            image1 = load_image(f).convert('RGB')
        with open(imagename2, "rb") as f:
            image2 = load_image(f).convert('RGB')

        with open(labelname1, "rb") as f:
            label1 = load_image(f).convert('L')
        with open(labelname2, "rb") as f:
            label2 = load_image(f).convert('L')

        if self.input_transform is not None:
            image1 = self.input_transform(image1)
            image2 = self.input_transform(image2)

        if self.label_transform is not None:
            label1 = self.label_transform(label1)
            label2 = self.label_transform(label2)

        return image1, image2, label1, label2

    def __len__(self):
        return len(self.val_names)
