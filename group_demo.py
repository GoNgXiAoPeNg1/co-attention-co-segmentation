import torch
from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Resize
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import PIL.Image as Image
import numpy as np
from model import *
from glob import glob
import argparse
from utils.utils import *
import cv2
import pdb
data_dir = '/home/gongxp/mlmr/gongxp/Keras-FCN-CosegData_train_shuangfenzhibeifen/datasets/' \
           'icoseg_internet/sub_Internet/Horse100/'
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_ids', default=[0,1], help='a list of gpus')
parser.add_argument('--image_path', default=data_dir + 'Images', type=str, help="image path")
parser.add_argument('--output_path',default='/home/gongxp/mlmr/githubcode/Semantic-Aware-Attention-Based-Deep-Object-Co-segmentation/model_path/group_res/', type=str, help="output path")
parser.add_argument('--model', default='/home/gongxp/mlmr/githubcode/'
                                       'Semantic-Aware-Attention-Based-Deep-Object-Co-segmentation/model_path/epoch10iter3200loss0.039302204.pkl',
                    type=str, help="model path")
args = parser.parse_args()


class Demo:
    def __init__(self):
        self.args = args
        self.net = model().cuda()
        self.net = nn.DataParallel(self.net, device_ids=self.args.gpu_ids)
        self.net.load_state_dict(torch.load(self.args.model))
        self.input_transform = Compose([Resize((512, 512)), ToTensor(
        ), Normalize([.485, .456, .406], [.229, .224, .225])])
        self.image_path = self.args.image_path

    def group_demo(self):
        self.net.eval()
        attentions=[]
        images_path=glob(self.image_path+"/*.jpg")
    
        for image_path in images_path:
            image=Image.open(image_path).convert('RGB')
            image = self.input_transform(image)
            image = image.unsqueeze(0).cuda()
            feature,attention =self.net.module.generate_attention(image)
            attentions.append(attention)

        group_mean_attentions=torch.stack(attentions)
        group_mean_attention=torch.mean(torch.stack(attentions),dim=0)

        for index,image_path in enumerate(images_path):
            image=Image.open(image_path).convert('RGB')
            image = self.input_transform(image)
            image = image.unsqueeze(0).cuda()
            feature,attention =self.net.module.generate_attention(image)

            features_map =self.net.module.features_map(image)
            spatial_avg_pool =self.net.module.spatial_avg_pool(features_map)
            mask=self.net.module.dec(feature*group_mean_attention*spatial_avg_pool)
            mask = torch.argmax(mask, dim=1)
            mask = as_numpy(mask.squeeze(0).cpu())
            cv2.imwrite(self.args.output_path+"c_%d.jpg"%(index), mask*255)
            # image = (image - image.min()) / image.max()
            # mask = torch.cat([torch.zeros(1, 512, 512).long().cuda(
            #         ), mask, torch.zeros(1, 512, 512).long().cuda()]).unsqueeze(0)
            # save_image(mask.float().data * 255 ,
            #        self.args.output_path+"co_%d.jpg"%(index), normalize=True)


if __name__ == "__main__":
    with torch.no_grad():
        demo = Demo()
        demo.group_demo()

print("Finish!!!")
