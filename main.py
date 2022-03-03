'''
Paper: An Effective Dual-model with Compact Imaging Network and Relative Histogram Redistribution for Underwater Image Enhancement
How to use: python test.py --checkpoint CHECKPOINTS_PATH
Instance: python test.py --checkpoint ./checkpoints/model_best.pth
'''
import os
import torch
import cv2
import natsort
import numpy as np
from PIL import Image
from model import PhysicalNN
import argparse
from torchvision import transforms
import torch.nn as nn
import datetime
import time
import math

from LabStretching import LABStretching
from color_equalisation import RGB_equalisation
from global_stretching_RGB import stretching
from relativeglobalhistogramstretching import RelativeGHstretching

def main(checkpoint, imgs_path, result_path):

    ori_dirs = []
    for image in os.listdir(imgs_path):
        ori_dirs.append(os.path.join(imgs_path, image))

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = PhysicalNN()
    model = torch.nn.DataParallel(model).to(device)
    print("loading trained model ...")
    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.module
    model.eval()

    testtransform = transforms.Compose([
                transforms.ToTensor(),
            ])
    unloader = transforms.ToPILImage()

    starttime = datetime.datetime.now()
    for imgdir in ori_dirs:
        img_name = (imgdir.split('/')[-1]).split('.')[0]
        img = Image.open(imgdir)
        inp = testtransform(img).unsqueeze(0)
        inp = inp.to(device)
        out = model(inp)

        corrected = unloader(out.cpu().squeeze(0))
        dir = '{}/results'.format(result_path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        corrected.save(dir+'/{}_primary_enhanced.jpg'.format(img_name))
    endtime = datetime.datetime.now()
    print('********    the first model is finished   ********')
    print('"CIN-model" time cost',endtime-starttime,'s')

if __name__ == '__main__':

###  CIN-model
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help='checkpoints path', required=True)
    parser.add_argument(
            '--images', help='test images folder', default='./test_images/')
    parser.add_argument(
            '--result', help='results folder', default='./pri_enhanced_results/')
    args = parser.parse_args()
    checkpoint = args.checkpoint
    imgs = args.images
    result_path = args.result
    main(checkpoint=checkpoint, imgs_path=imgs, result_path=result_path)

###  RHR-model
    path = "./pri_enhanced_results/results/"
    path_final = "./final_enahnced_results/"
    files = os.listdir(path)
    files =  natsort.natsorted(files)
    time_start=time.time()
    for i in range(len(files)):
        file = files[i]
        filepath = path + "/" + file
        prefix = file.split('.')[0]
        if os.path.isfile(filepath):
            img = cv2.imread(path + file)
            print('processing',file)
            # img = cv2.imread('InputImages/' + file)
            # path = np.unicode(path, 'utf-8')
            # img = cv2.imread('InputImages/' + file)
            #img = cv2.imread(np.unicode(folder +'/results_2842_compensate0.3/' + file, 'utf-8'))
            #height = len(img)
            #width = len(img[0])
            sceneRadiance = img
            #sceneRadiance = RGB_equalisation(img)
            #sceneRadiance = RelativeGHstretching(sceneRadiance, height, width)
            sceneRadiance = stretching(sceneRadiance)
            sceneRadiance = LABStretching(sceneRadiance)
            if not os.path.exists(path_final):
                os.makedirs(path_final)
            cv2.imwrite(path_final + prefix + '_finally_enhanced.jpg', sceneRadiance)
    time_end=time.time()
    print('********    the second model is finished   ********')
    print('"RHR-model"time cost',time_end-time_start,'s')
