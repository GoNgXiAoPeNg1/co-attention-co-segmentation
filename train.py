import torch
import torchvision
from torchvision.models import vgg16
from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Resize, ToTensor, ToPILImage
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
import skimage.io as io
from torchsummary import summary
import numpy as np
import glob
import cv2
import itertools
import PIL.Image as Image
import argparse
import pdb
from datasets import coseg_val_dataset, coseg_train_dataset
from model import *
 
from utils.utils import *
from model_siamese import crate_Den_Resnet_model
from model_dropout import UnetResNet as UnetResNet_dropout
from model_siamese_16x import UnetResNet as model_siamese_16x
from model_siamese_addbn import UnetResNet as UnetResNetaddbn
from model_siamese_addbn_cor import UnetResNet as UnetResNetaddbn_cor
from model_siamase_groupnor import UnetResNet as UnetResNet_groupnor

from model_siamese_addbn_advance import UnetResNet as model_siamese_addbn_advance
from model_siamese_addbn_pooling_sort import UnetResNet as model_siamese_addbn_pooling_sort
from model_siamese_addbn_advance_pooling import UnetResNet as model_siamese_addbn_advance_pooling
from model_siamese_addbn_w_x_pooling import UnetResNet as model_siamese_addbn_w_x_pooling
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.nn.Module.dump_patches = True
# input arguments
parser = argparse.ArgumentParser(description='Attention Based Co-segmentation')
parser.add_argument('--lr', default=6e-4, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=0.00005,
                    help='weight decay value')
parser.add_argument('--gpu_ids', default=[0], help='a list of gpus')
parser.add_argument('--num_worker', default=4, help='numbers of worker')
parser.add_argument('--batch_size', default=4, help='bacth size')
parser.add_argument('--epoches', default=100, help='epoches')

datapath = '/media/gongxp/2eae26dd-69c7-4195-94af-3604072e182f/gxp/paperdata/combine_MSRC_VOC2012_most_gray_Pos_Neg/'
datapathin_ico = '/media/gongxp/2eae26dd-69c7-4195-94af-3604072e182f/gxp/internet_icoseg/'
# train_1800.txt  val_230.txt
# train13200.txt   val857.txt
# horse_test.txt
# datapath = '/home/gongxp/mlmr/gongxp/Keras-FCN-CosegData_train_shuangfenzhibeifen/datasets/combine_MSRC_VOC2012_most_gray_Pos_Neg/'
parser.add_argument('--train_data', default=datapath+'origin_image/', help='training data directory')
parser.add_argument('--val_data', default=datapath+'origin_image/', help='validation data directory')
parser.add_argument('--train_txt', default=datapath+'train_1800.txt', help='training image pair names txt')
parser.add_argument('--val_txt', default=datapath+'val_230.txt', help='validation image pair names txt')
parser.add_argument('--train_label', default=datapath+'ground_truth/', help='training label directory')
parser.add_argument('--val_label', default=datapath+'ground_truth/', help='validation label directory')
parser.add_argument('--model_path', default="model_path/fewdata_512_epoch5loss1.5476iou0.5064.ckpt", help='model saving directory')
parser.add_argument('--model_save_path', default="model_path/", help='model saving directory')

args = parser.parse_args()

def one_hot(scores, labels):
    labels = torch.unsqueeze(labels, 1)
    _labels = torch.zeros_like(scores)

    _labels.scatter_(dim=1, index=labels, value=1)#scatter_  带下划线的函数 是inplace  ，就是内部赋值，没有返回值
    _labels.requires_grad = False
    return _labels
def get_file_len(file_path):
    fp = open(file_path)
    lines = fp.readlines()
    fp.close()
    return len(lines)
class Relabel:
    def __call__(self, tensor):
        assert isinstance(
            tensor, torch.LongTensor), 'tensor needs to be LongTensor'
        tensor[tensor > 0] = 1
        return tensor

# numpy -> tensor


class ToLabel:
    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long()



class Trainer:
    def __init__(self):
        self.args = args

        self.input_transform = Compose([Resize((512,512)), ToTensor( # Resize((resize,resize)),
        ), Normalize([.485, .456, .406], [.229, .224, .225])])
        self.label_transform = Compose(
            [Resize((512,512)),ToLabel(),Relabel()])

        # self.net = model().cuda()
        # 相关性改成这种
        self.net = crate_Den_Resnet_model().cuda()
        # self.net = UnetResNetaddbn().cuda()
        # self.net = UnetResNet_dropout().cuda()
        # self.net = model_siamese_addbn_advance().cuda()
        # self.net = model_siamese_addbn_pooling_sort().cuda()
        # self.net = model_siamese_addbn_advance_pooling().cuda()
        # self.net = model_siamese_addbn_w_x_pooling().cuda()
        # checkpoint = torch.load(self.args.model_path)
        # self.net.load_state_dict(checkpoint,strict=True)
        #
        # self.net = torch.load(self.args.model_path).cuda()

        # checkpoint = torch.load('/home/gongxp/mlmr/githubcode/siamase_pytorch/resnet50_origin.pth')
        # self.net.load_state_dict(checkpoint, strict=False)
        self.train_data_loader = DataLoader(coseg_train_dataset(self.args.train_data, self.args.train_label, self.args.train_txt, self.input_transform, self.label_transform),
                                            num_workers=self.args.num_worker, batch_size=self.args.batch_size, shuffle=True)
        self.val_data_loader = DataLoader(coseg_val_dataset(self.args.val_data, self.args.val_label, self.args.val_txt, self.input_transform, self.label_transform),
                                          num_workers=self.args.num_worker, batch_size=self.args.batch_size, shuffle=False)
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        # self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=0.9, nesterov=True,weight_decay=self.args.weight_decay)
        #
        self.steps_per_epoch = int(np.ceil(get_file_len(self.args.train_txt) / float(self.args.batch_size)))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.steps_per_epoch * 2, gamma=0.75)

        self.loss_func = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss2d()  # error

        self.BCEsoftJaccarddice = BCESoftJaccardDice()  # error
        self.BCESoftJaccarddice_rate_change = BCESoftJaccardDiceRateChange()  # error
        # self.dice_loss = DiceLoss()  #
        summary(self.net, [(3, 512, 512), (3, 512, 512)])

    def pixel_accuracy(self, output, label):
        correct = len(output[output == label])
        wrong = len(output[output != label])
        return correct, wrong

    def jaccard(self, output, label):

        temp = output[label == 1]
        i = len(temp[temp == 1])
        temp = output + label
        u = len(temp[temp > 0])
        return i, u

    def precision(self, output, label):
        temp = output[label == 1]
        tp = len(temp[temp == 1])
        p = len(output[output > 0])
        return tp, p

    def evaluate(self, net, epoch):
        self.net.eval()
        correct = 0
        wrong = 0
        intersection = 0
        union = 0
        true_positive = 0
        positive = 1
        for i, (image1, image2, label1, label2) in enumerate(self.val_data_loader):

            image1, image2, label1, label2 = image1.cuda(
            ), image2.cuda(), label1.cuda(), label2.cuda()
            output1, output2 = self.net(image1, image2)
            output1 = torch.argmax(output1, dim=1)
            output2 = torch.argmax(output2, dim=1)
            # eval output1
            c, w = self.pixel_accuracy(output1, label1)
            correct += c
            wrong += w

            ii, u = self.jaccard(output1, label1)

            intersection += ii
            union += u

            tp, p = self.precision(output1, label1)
            true_positive += tp
            positive += p
            # eval output2
            c, w = self.pixel_accuracy(output2, label2)
            correct += c
            wrong += w

            ii, u = self.jaccard(output2, label2)
            intersection += ii
            union += u

            tp, p = self.precision(output2, label2)
            true_positive += tp
            positive += p

        print("pixel accuracy: {} correct: {}  wrong: {}".format(
            correct / (correct + wrong), correct, wrong))
        print("precision: {} true_positive: {} positive: {}".format(
            true_positive / positive, true_positive, positive))
        print("jaccard score: {} intersection: {} union: {}".format(
            intersection / union, intersection, union))
        self.net.train()
        return correct / (correct + wrong), intersection / union, true_positive / positive
        # return intersection / union

    def train(self):
        iou_last = 0
        for epoch in range(self.args.epoches):
            losses = []

            print('current lr: ', round(self.optimizer.param_groups[0]['lr'], 10))
            for i, (image1, image2, label1, label2) in enumerate(self.train_data_loader):

                image1, image2, label1, label2 = image1.cuda(
                ), image2.cuda(), label1.cuda(), label2.cuda()

                output1, output2 = self.net(image1, image2)
                # calculate loss from output1 and output2
                # loss_func + BCESoftJaccarddice_rate_change
                loss = self.loss_func(output1, label1)
                loss += self.loss_func(output2,  label2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                losses.append(loss.data.cpu().numpy())

                if i % 10 == 0:
                    print("---------------------------------------------")
                    print(
                        "epoch{} iter {}/{} BCE loss{}:".format(epoch, i, len(self.train_data_loader), np.mean(losses)))

                if i % 100 == 0:
                    print("testing......")
                    acc, jac, pre = self.evaluate(self.net, epoch)
                    print("testing over")

            if iou_last<jac:
                iou_last = jac
                print("saving......")
                torch.save(self.net,self.args.model_save_path + 'k==0.75_spatial_notop_model_512_cor_epoch' + str(epoch) + 'loss' +
                           str(round(np.mean(losses),4)) + 'iou' + str(round(jac,4)) + '.ckpt')


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
