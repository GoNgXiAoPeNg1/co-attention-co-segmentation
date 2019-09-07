import torch
import PIL.Image as Image
from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, Scale, ToPILImage
from torch.utils.data import DataLoader
import numpy as np
from torchvision.utils import save_image
import argparse
from model import *
import pdb
import collections
from utils.inference import *
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser()
data_dir = '/home/gongxp/mlmr/gongxp/Keras-FCN-CosegData_train_shuangfenzhibeifen/datasets/' \
           'icoseg_internet/sub_Internet/'

parser.add_argument('--gpu_ids', default=[1], help='a list of gpus')
parser.add_argument('--nb_classes', default=2, help='nb_classes')
parser.add_argument('--image_path', default=data_dir + 'Airplane100/Images', type=str, help="image path")
parser.add_argument('--output_bin', default='/home/gongxp/mlmr/githubcode/'
                                            'Semantic-Aware-Attention-Based-Deep-Object-Co-segmentation/model_path/res1/',
                    type=str, help="output1")
parser.add_argument('--output_color', default='/home/gongxp/mlmr/githubcode/'
                                              'Semantic-Aware-Attention-Based-Deep-Object-Co-segmentation/model_path/res_color/',
                    type=str, help="output")
parser.add_argument('--label_path', default=data_dir + 'Airplane100/GroundTruth', type=str, help="label")

args = parser.parse_args()

def calculate_iou(nb_classes, res_dir, label_dir, data_dir, image_list):
    conf_m = np.zeros((nb_classes, nb_classes), dtype=float)
    total = 0
    for img_name in image_list:
        img_name = img_name.strip('\n')
        total += 1
        print('#%d: %s' % (total, img_name))
        res_img = Image.open('%s%s' % (res_dir, img_name.replace('jpg', 'png')))

        pred = np.asarray(res_img).astype(int)
        label = np.asarray(Image.open('%s/%s' % (label_dir, img_name.replace('jpg', 'png')))).astype(
            int)  # np.array([])#
        flat_pred = np.ravel(pred)

        flat_pred = np.array((flat_pred), dtype=int)
        flat_label = np.ravel(label)
        flat_label[np.where(flat_label!=0)] = 1
        for p, l in zip(flat_pred, flat_label):
            if l == 255:
                continue
            if l < nb_classes and p < nb_classes:
                conf_m[l, p] += 1
            else:
                print('Invalid entry encountered, skipping! Label: ', l,
                      ' Prediction: ', p, ' Img_num: ', img_name)

    I = np.diag(conf_m)
    U = np.sum(conf_m, axis=0) + np.sum(conf_m, axis=1) - I
    IOU = I / U
    meanIOU = np.mean(IOU)
    return conf_m, IOU, meanIOU


class Demo:
    def __init__(self, net):
        self.args = args

        # self.net = model().cuda()
        self.net = net
        # self.net = nn.DataParallel(self.net, device_ids=self.args.gpu_ids)

        self.input_transform = Compose([Resize((512, 512)), ToTensor(
        ), Normalize([.485, .456, .406], [.229, .224, .225])])
        self.image_path = self.args.image_path
        self.label_path = self.args.label_path

    def single_demo(self):
        self.net.eval()
        current_dir = os.path.dirname(os.path.realpath(__file__))

        image_list = os.listdir(self.args.image_path)
        image_list.sort()
        start_time = time.time()
        inference(self.input_transform, self.net, image_list, self.args.image_path, save_dir=self.args.output_bin)
        duration = time.time() - start_time
        print('{}s used to make predictions.\n'.format(duration))

        conf_m, IOU, meanIOU = calculate_iou(self.args.nb_classes, self.args.output_bin, self.args.label_path, self.args.image_path,
                                             image_list)
        print('IOU: ')
        print(IOU)
        print('meanIOU: %f' % meanIOU)
        print('pixel acc: %f' % (np.sum(np.diag(conf_m)) / np.sum(conf_m)))
        return meanIOU, np.sum(np.diag(conf_m)) / np.sum(conf_m)


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

    def evaluate(self, output1, output2, label1, label2):
        correct = 0
        wrong = 0
        intersection = 0
        union = 0
        true_positive = 0
        positive = 1
        # pdb.set_trace()
        label1, label2 = np.asarray(label1) / 255, np.asarray(label2) / 255

        output1 = torch.argmax(output1, dim=1)
        output2 = torch.argmax(output2, dim=1)

        output1 = as_numpy(output1.squeeze(0).cpu())
        output2 = as_numpy(output2.squeeze(0).cpu())
        # output1 = np.expand_dims(output1, -1)

        output1 = Image.fromarray(np.uint8(output1), mode='L')
        output1 = output1.resize(self.origin_size1)
        output1 = np.asarray(output1)

        output2 = Image.fromarray(np.uint8(output2), mode='L')
        output2 = output2.resize(self.origin_size2)
        output2 = np.asarray(output2)

        # cv2.imwrite('sss.png', output1 * 255)
        # output1 = output1.resize((self.origin_size1[1], self.origin_size1[0]))
        # output2 = output2.resize( (self.origin_size2[1], self.origin_size2[0]))
        # eval output1
        c, w = self.pixel_accuracy(output1, label1)
        correct += c
        wrong += w

        i, u = self.jaccard(output1, label1)
        intersection += i
        union += u

        tp, p = self.precision(output1, label1)
        true_positive += tp
        positive += p
        # eval output2
        c, w = self.pixel_accuracy(output2, label2)
        correct += c
        wrong += w

        i, u = self.jaccard(output2, label2)
        intersection += i
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


def main(net):
    demo = Demo(net)
    return demo.single_demo()

print("Finish!!!")
