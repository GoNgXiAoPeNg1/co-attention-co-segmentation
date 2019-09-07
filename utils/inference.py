import os
from math import ceil
import numpy as np
import pdb
import time
# from pylab import *
# import sys
import cv2
from PIL import Image
from utils.utils import *

def pairwise(iterable):
    a = iter(iterable)
    return zip(a, a)

def inference(input_transform, net, image_list, data_dir, save_dir):
    save_dir = save_dir

    total = 0
    if len(image_list) % 2 != 0:
        image_list.append(image_list[0])
    for img_num1, img_num2 in pairwise(image_list):
        img_num1 = img_num1.strip('\n')
        img_num2 = img_num2.strip('\n')
        total += 1
    
        print('#%d: %s,%s' % (total, img_num1, img_num2))
        image1 = Image.open('%s/%s' % (data_dir, img_num1)).convert('RGB')
        image2 = Image.open('%s/%s' % (data_dir, img_num2)).convert('RGB')
        origin_size1 = image1.size
        origin_size2 = image2.size

        # image1 = np.asarray(image1)  # , data_format='default')
        # image2 = np.asarray(image2)

        start_time = time.time()

        image1 = input_transform(image1)
        image2 = input_transform(image2)
        image1, image2 = image1.unsqueeze(0).cuda(), image2.unsqueeze(0).cuda()

        output1, output2 = net(image1, image2)

        duration = time.time() - start_time
        print('{}s used to make predict two pics.\n'.format(duration))

        output1 = torch.argmax(output1, dim=1)
        output2 = torch.argmax(output2, dim=1)

        output1 = np.asarray(output1.squeeze(0).cpu())*255
        output2 = np.asarray(output2.squeeze(0).cpu())*255



        # cv2.imwrite(os.path.join(save_dir, img_num1.replace('jpg', 'png')), output1*255)
        output1 = Image.fromarray(np.uint8(output1), mode='L')
        # output1 = output1.resize(origin_size1,Image.BILINEAR)

        output2 = Image.fromarray(np.uint8(output2), mode='L')
        # output2 = output2.resize(origin_size2,Image.BILINEAR)

        output1 = Image.fromarray(np.uint8(output1), mode='L')
        output2 = Image.fromarray(np.uint8(output2), mode='L')
        output1 = np.asarray(output1)
        output2 = np.asarray(output2)
        output1 = cv2.GaussianBlur(output1, (1, 1), 2)
        output2 = cv2.GaussianBlur(output2, (1, 1), 2)

        output1 = Image.fromarray(np.uint8(output1), mode='L')
        output2 = Image.fromarray(np.uint8(output2), mode='L')
        output1 = output1.resize(origin_size1)
        output2 = output2.resize(origin_size2)
        if save_dir:
            output1.save(os.path.join(save_dir, img_num1.replace('jpg', 'png')))
            output2.save(os.path.join(save_dir, img_num2.replace('jpg', 'png')))
