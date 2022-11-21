import time
import torch
import torch.nn.functional as F
import sys
import numpy as np
import os, argparse
import cv2
from the_net import Baseline
from data import test_dataset
import numpy as np
from torchvision import utils,transforms
from PIL import Image
import cv2
from models import *
from thop import profile
from torchstat import stat
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
parser = argparse.ArgumentParser()
parser.add_argument('--parameter',  default='GIE_MODEL', help='name of parameter file')
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--gpu_id', type=str, default='2', help='select gpu id')
# parser.add_argument('--test_path',type=str,default='Dataset/VT5000/VT5000_clear/',help='test dataset path')
# parser.add_argument('--test_path',type=str,default='Dataset/VT1000/VT1000/',help='test dataset path')
parser.add_argument('--test_path',type=str,default='Dataset/VT821/VT821/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
if opt.gpu_id=='5':
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    print('USE GPU 5')
elif opt.gpu_id=='2':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    print('USE GPU 2')

#load the model
model = Baseline()
model.load_state_dict(torch.load('the_model/TNet.pth'))
model.cuda()
model.eval()

#GIE
net = Net().cuda()
net.load_state_dict(torch.load(opt.parameter))
net.eval()


#test
#for 821,1000
test_datasets = ['']
#for 5000
# test_datasets = ['Test']

for dataset in test_datasets:
    save_path = 'Results/821/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    t_root=dataset_path +dataset +'/T/'
    test_loader = test_dataset(image_root, gt_root,t_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt,t, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        t = t.cuda()
        with torch.no_grad():
            R, L = net(image)
        
        res,ttt,u4,u3,u2,u1= model(image,t,L)

        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('save img to: ',save_path+name)
        cv2.imwrite(save_path+name,res*255)
    print('Test Done!')