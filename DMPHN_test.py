import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import math
import argparse
import random
import models
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import time
from PIL import Image

#Hyper Parameters
METHOD = "DMPHN_1_2_4"

SAMPLE_DIR = "./new_dataset/val/HAZY"    

EXPDIR = 'DMPHN_results'

GPU = 0


def save_images(images, name):
    filename = './test_results/' + EXPDIR + "/" + name
    torchvision.utils.save_image(images, filename)


def main():
    print("init data folders")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_lv1 = models.Encoder().to(device)
    encoder_lv2 = models.Encoder().to(device)
    encoder_lv3 = models.Encoder().to(device)

    decoder_lv1 = models.Decoder().to(device)
    decoder_lv2 = models.Decoder().to(device)
    decoder_lv3 = models.Decoder().to(device)


    encoder_lv1.load_state_dict(torch.load(str('./checkpoints3/' + METHOD + "/encoder_lv1.pkl"), map_location=torch.device('cpu')))
    encoder_lv2.load_state_dict(torch.load(str('./checkpoints3/' + METHOD + "/encoder_lv2.pkl"), map_location=torch.device('cpu')))
    encoder_lv3.load_state_dict(torch.load(str('./checkpoints3/' + METHOD + "/encoder_lv3.pkl"), map_location=torch.device('cpu')))

    decoder_lv1.load_state_dict(torch.load(str('./checkpoints3/' + METHOD + "/decoder_lv1.pkl"), map_location=torch.device('cpu')))
    decoder_lv2.load_state_dict(torch.load(str('./checkpoints3/' + METHOD + "/decoder_lv2.pkl"), map_location=torch.device('cpu')))
    decoder_lv3.load_state_dict(torch.load(str('./checkpoints3/' + METHOD + "/decoder_lv3.pkl"), map_location=torch.device('cpu')))

    os.makedirs('./test_results/' + EXPDIR, exist_ok = True)

    iteration = 0.0
    test_time = 0.0
    
    for images_name in os.listdir(SAMPLE_DIR):
        # print (images_name )
        with torch.no_grad():             
            images_lv1 = transforms.ToTensor()(Image.open(SAMPLE_DIR + '/' + images_name).convert('RGB'))
            images_lv1 = Variable(images_lv1 - 0.5).unsqueeze(0).to(device)
            # images_lv1 = Variable(images_lv1 - 0.5).unsqueeze(0)

            start = time.time()
            H = images_lv1.size(2)
            W = images_lv1.size(3)

            images_lv2_1 = images_lv1[:,:,0:int(H/2),:]
            images_lv2_2 = images_lv1[:,:,int(H/2):H,:]
            images_lv3_1 = images_lv2_1[:,:,:,0:int(W/2)]
            images_lv3_2 = images_lv2_1[:,:,:,int(W/2):W]
            images_lv3_3 = images_lv2_2[:,:,:,0:int(W/2)]
            images_lv3_4 = images_lv2_2[:,:,:,int(W/2):W]

            feature_lv3_1 = encoder_lv3(images_lv3_1)
            feature_lv3_2 = encoder_lv3(images_lv3_2)
            feature_lv3_3 = encoder_lv3(images_lv3_3)
            feature_lv3_4 = encoder_lv3(images_lv3_4)
            feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3)
            feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3)
            feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
            residual_lv3_top = decoder_lv3(feature_lv3_top)
            residual_lv3_bot = decoder_lv3(feature_lv3_bot)

            feature_lv2_1 = encoder_lv2(images_lv2_1 + residual_lv3_top)
            feature_lv2_2 = encoder_lv2(images_lv2_2 + residual_lv3_bot)
            feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3
            residual_lv2 = decoder_lv2(feature_lv2)

            feature_lv1 = encoder_lv1(images_lv1 + residual_lv2) + feature_lv2
            dehazed_image = decoder_lv1(feature_lv1)
        
            stop = time.time()
            test_time += stop-start
            print('RunTime:%.4f'%(stop-start), '  Average Runtime:%.4f'%(test_time/(iteration+1)))
            save_images(dehazed_image.data + 0.5, images_name) 
   
            
if __name__ == '__main__':
    main()

        

        

