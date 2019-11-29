import argparse
import h5py
import random
import os
import numpy as np
import pickle
import glob
import cv2
from PIL import Image as pil_image

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn
import torchvision.transforms as transforms

from models import UNet, UNet11, UNet16, UNet16BN, LinkNet34

class TestDataset(Dataset):
    def __init__(self, image_ids, image_path, transform=None):
        self.image_ids = image_ids
        self.image_path = image_path
        self.transform = transform
        self.n = len(image_ids)

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        ### load image
        image_file = self.image_path + '/%s.jpg' % img_id
        img_np, W, H = load_image_from_file(image_file)

        #Image.resize(size, resample=0), PIL.Image.NEAREST
        ######
        return img_id, img_np, W, H

def load_image_from_file(image_file):
    img = pil_image.open(image_file)
    img = img.convert('RGB')
    img_np = np.asarray(img, dtype=np.float)
    ### why only 0-255 integers
    img_np = (img_np / 255.0).astype('float32')
    ### resize the image
    #img_np = cv2.resize(img_np, (512, 512), interpolation=cv2.INTER_CUBIC)
    if len(img_np.shape) == 2:
        img_np = img_np[:, :, np.newaxis]
    (H, W, C) = img_np.shape
    # print(img_np.shape)
    img_np = cv2.resize(img_np, (512, 512), interpolation=cv2.INTER_CUBIC)

    return img_np, W, H


def load_image_from_h5(image_file):
    f = h5py.File(image_file, 'r')
    img_np = f['img'].value
    img_np = (img_np / 255.0).astype('float32')

    return img_np



def load_image_from_file_and_save_to_h5(img_id, image_file, temp_path, resize=True):
    if resize:
        img = load_img(image_file, target_size=(512,512), grayscale=False)  # this is a PIL image
    else:
        img = load_img(image_file, grayscale=False)  # this is a PIL image
    img_np = img_to_array(img)
    ### why only 0-255 integers
    save_path = temp_path
    img_np = img_np.astype(np.uint8)
    hdf5_file = h5py.File(save_path + '%s_W%s_H%s.h5' % (img_id,img_np.shape[0],img_np.shape[1]), 'w')
    hdf5_file.create_dataset('img', data=img_np, dtype=np.uint8)
    hdf5_file.close()
    return img_np


def test_new_data(model_weight, image_path, temp_path, output_path, model):

    image_ids = sorted([fname.split('/')[-1].split('.')[0] for fname in glob.glob(os.path.join(image_path, '*.jpg'))])
    #if len(image_ids) == 0:
    #    print('No image found')

    data_set = TestDataset(image_ids, image_path)
    test_loader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=10, pin_memory=False)

    if model == 'UNet16':
        model = UNet16(num_classes=5, pretrained='vgg')
    elif model == 'UNet16BN':
        model = UNet16BN(num_classes=5, pretrained='vgg')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ## my model was trained on multi-GPUs
    ## need to wrap it with nn.DataParallel
    model = nn.DataParallel(model)
    model.to(device)
    print('load model weight')
    state = torch.load(model_weight)
    model.load_state_dict(state['model'])

    cudnn.benchmark = True
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    attr_types = ['pigment_network', 'negative_network', 'streaks', 'milia_like_cyst', 'globules']
    with torch.no_grad():
        for img_id, test_image, W, H in test_loader:
            img_id = img_id[0]
            W = W[0].item()
            H = H[0].item()
            print('Loading', img_id, 'W',W, 'H',H, 'resized image',test_image.size())
            test_image = test_image.to(device)  # [N, 1, H, W]
            test_image = test_image.permute(0, 3, 1, 2)
            outputs, outputs_mask_ind1, outputs_mask_ind2 = model(test_image)
            test_prob = torch.sigmoid(outputs)
            test_prob = test_prob.squeeze().data.cpu().numpy()
            for ind, attr in enumerate(attr_types):
                resize_mask = cv2.resize(test_prob[ind, :, :], (W, H), interpolation=cv2.INTER_CUBIC)
                #for cutoff in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                for cutoff in [0.3]:
                    #if not os.path.exists('submission_%s'%(cutoff)): os.makedirs('submission_%s'%(cutoff))
                    test_mask = (resize_mask>cutoff).astype('int') *255.0
                    cv2.imwrite(os.path.join(output_path,"ISIC_%s_attribute_%s.png" % (img_id.split('_')[1],attr)), test_mask)
            #break

## cutoff validation jaccard
## 0.0 0.012
## 0.1 0.475
## 0.2 0.476
## 0.3 0.477
## 0.4 0.477
## 0.5 0.477
def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', type=str, default='UNet16', choices=['UNet', 'UNet11', 'UNet16', 'UNet16BN', 'LinkNet34'])
    arg('--model-weight', type=str, default=None)
    arg('--image-path', type=str, default='data', help='image path')
    arg('--temp-path', type=str, default='temp', help='temporary folder for preprocessed data')
    arg('--output-path', type=str, default='prediction', help='prediction')
    
    args = parser.parse_args()
    
    model = args.model
    model_weight = args.model_weight
    if model_weight is None:
        raise ValueError('Please specify model-weight')


    image_path = args.image_path
    nfiles = len(glob.glob(os.path.join(image_path, '*.jpg')))
    if nfiles == 0 :
        raise ValueError('No images found')
    else:
        print('%s images found' % nfiles)
    
    temp_path = args.temp_path
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    
    test_new_data(model_weight, image_path, temp_path, output_path, model)



if __name__ == '__main__':
    main()