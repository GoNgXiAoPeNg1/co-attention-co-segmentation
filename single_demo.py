import torch
import PIL.Image as Image
from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, Scale, ToPILImage,Pad
from torch.utils.data import DataLoader
import numpy as np
from torchvision.utils import save_image
import argparse
from model_siamese_addbn import UnetResNet as UnetResNetaddbn
from torchsummary import summary
# from model import *
# from model_rnn import *

from model_siamese import crate_Den_Resnet_model
import pdb
import collections
from utils.inference import *
import time
from tensorboardX import SummaryWriter
writer = SummaryWriter('visualkernal/')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
data_dir = '/home/gongxp/mlmr/gongxp/Keras-FCN-CosegData_train_shuangfenzhibeifen/datasets/' \
           'icoseg_internet/sub_Internet/'
# data_dir ='/media/gongxp/2eae26dd-69c7-4195-94af-3604072e182f/gxp/paperdata/icoseg_internet/sub_Internet_Neg/'
class_ = 'Airplane100/'
#  Car100 Horse100
data_dir_test = '/media/gongxp/2eae26dd-69c7-4195-94af-3604072e182f/gxp/internet_icoseg/'
# data_dir1 ='/media/gongxp/2eae26dd-69c7-4195-94af-3604072e182f/gxp/dataset1/'
parser.add_argument('--gpu_ids', default=[1], help='a list of gpus')
parser.add_argument('--nb_classes', default=2, help='nb_classes')
parser.add_argument('--image_path', default=data_dir_test+'origin_image/', type=str, help="image path")
parser.add_argument('--label_path', default=data_dir_test+ 'ground_truth/', type=str, help="label1")



# data_dir ='/media/gongxp/2eae26dd-69c7-4195-94af-3604072e182f/gxp/dataset/test/vr40/'
# test_data_dir = '/home/gongxp/mlmr/githubcode/segment_hybird/test_image/'
# # data_dir1 ='/media/gongxp/2eae26dd-69c7-4195-94af-3604072e182f/gxp/dataset1/'
# parser.add_argument('--gpu_ids', default=[1], help='a list of gpus')
# parser.add_argument('--nb_classes', default=2, help='nb_classes')
# parser.add_argument('--image_path', default=data_dir+'origin-image', type=str, help="image path")
# parser.add_argument('--label_path', default=data_dir + 'ground-truth', type=str, help="label1")


parser.add_argument('--output_bin', default='/home/gongxp/mlmr/githubcode/'
                                            'Semantic-Aware-Attention-Based-Deep-Object-Co-segmentation/model_path/res/',
                    type=str, help="output1")
# parser.add_argument('--output_bin', default='/media/gongxp/2eae26dd-69c7-4195-94af-3604072e182f/gxp/gt_result_256480/',
#                     type=str, help="output1")

parser.add_argument('--output_color', default='/home/gongxp/mlmr/githubcode/'
                'Semantic-Aware-Attention-Based-Deep-Object-Co-segmentation/model_path/res_color/',type=str, help="output")

parser.add_argument('--model', default='/home/gongxp/mlmr/githubcode/'
        'Semantic-Aware-Attention-Based-Deep-Object-Co-segmentation/model_path/inter_ico_crate_Den_Resnet_model_cor_512_epoch86loss0.5727iou0.9499.ckpt',
                    type=str, help="model path")
args = parser.parse_args()


def jaccard(output, label):
    temp = output[label == 1]
    i = len(temp[temp == 1])
    temp = output + label
    u = len(temp[temp > 0])
    return i, u
def calculate_iou(nb_classes, res_dir, label_dir, data_dir, image_list):
    conf_m = np.zeros((nb_classes, nb_classes), dtype=float)
    total = 0
    i, u = 0, 0
    for img_name in image_list:
        img_name = img_name.strip('\n')
        total += 1
        print('#%d: %s' % (total, img_name))

        res_img = Image.open('%s%s' % (res_dir, img_name.replace('jpg', 'png')))

        tmp_res_img = np.asarray(res_img)

        res_img.save(os.path.join(res_dir, img_name.replace('jpg', 'png')))
        # get_color_image
        data = cv2.imread(os.path.join(data_dir, img_name))
        data = Image.fromarray(np.uint8(data), mode='RGB')

        data = np.asarray(data)
        fg = cv2.bitwise_and(data, data, mask=tmp_res_img)  # foreground
        cv2.imwrite(os.path.join(args.output_color, img_name.replace('jpg', 'png')), fg)

        res_img = res_img.convert('1')
        pred = np.asarray(res_img).astype(int)

        flat_pred = np.ravel(pred)
        flat_pred = np.array((flat_pred), dtype=int)

        label = np.asarray(Image.open('%s/%s' % (label_dir, img_name.replace('jpg', 'png')))).astype(int)  # np.array([])#
        flat_label = np.ravel(label)
        flat_label[flat_label != 0] = 1

        for p, l in zip(flat_pred, flat_label):

            if l < nb_classes and p < nb_classes:
                conf_m[l, p] += 1
            else:
                print('Invalid entry encountered, skipping! Label: ', l,
                      ' Prediction: ', p, ' Img_num: ', img_name)
    I = np.diag(conf_m)
    U = np.sum(conf_m, axis=0) + np.sum(conf_m, axis=1) - I
    IOU = I / U
    return conf_m, IOU
    # return conf_m, [1,1]


class Demo:
    def __init__(self):
        resize = 512
        cropsize  = 512
        self.args = args
        # self.net = model().cuda()
        # self.net = UnetResNetaddbn().cuda()
        # self.net = crate_Den_Resnet_model().cuda()
        # # self.net = nn.DataParallel(self.net, device_ids=self.args.gpu_ids)
        # self.net.load_state_dict(torch.load(self.args.model), strict=False)

        self.net = torch.load(self.args.model).cuda()
        self.input_transform = Compose([Resize((512,512)),  ToTensor(     # Resize((resize,resize)),
        ), Normalize([.485, .456, .406], [.229, .224, .225])])
        self.image_path = self.args.image_path
        self.label_path = self.args.label_path

    def single_demo(self):
        self.net.eval()

        current_dir = os.path.dirname(os.path.realpath(__file__))

        image_list = os.listdir(self.args.image_path)
        image_list.sort()
        start_time = time.time()
        inference(self.input_transform, self.net, image_list, self.args.image_path,  save_dir=self.args.output_bin)
        duration = time.time() - start_time
        print('{}s used to make predictions.\n'.format(duration))

        conf_m, IOU = calculate_iou(self.args.nb_classes, self.args.output_bin, self.args.label_path, self.args.image_path,
                                             image_list)
        print('IOU: ')
        print(IOU)
        print('meanIOU: %f' % np.mean(IOU))
        print('pixel acc: %f' % (np.sum(np.diag(conf_m)) / np.sum(conf_m)))

if __name__ == "__main__":
    demo = Demo()
    demo.single_demo()

print("Finish!!!")
