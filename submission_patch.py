import h5py
import random
import os
import numpy as np
import pickle
import glob
import cv2
#from keras.preprocessing.image import img_to_array, load_img
from PIL import Image as pil_image

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

from models import UNet, UNet11, UNet16, LinkNet34

class TestDataset(Dataset):
    def __init__(self, image_ids, image_path, step = 100):
        self.image_ids = image_ids
        self.image_path = image_path
        self.image_file = self.image_path + '%s.jpg' % img_id
        self.img_np, self.W, self.H = load_image_from_file(self.image_file)
        self.step = step
        W_list1 = range(0, W-512, self.step)
        W_list2 = range(W-512, 0, self.step*(-1))


    def __len__(self):
        """
        This function gets called with len()

        1. The length should be a deterministic function of some instance variables and should be a non-ambiguous representation of the total sample count. This gets tricky especially when certain samples are randomly generated, be careful
        2. This method should be O(1) and contain no heavy-lifting. Ideally, just return a pre-computed variable during the constructor call.
        3. Make sure to override this method in further derived classes to avoid unexpected samplings.
        """
        return self.n

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        ### load image
        image_file = self.image_path + '%s.jpg' % img_id
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
    # img_np = cv2.resize(img_np, (512, 512), interpolation=cv2.INTER_CUBIC)
    return img_np, W, H

def test_new_data(model_weight, image_path):

    image_ids = sorted([fname.split('/')[-1].split('.')[0] for fname in glob.glob(image_path + '*.jpg')])
    if len(image_ids) == 0:
        print('No image found')

    data_set = TestDataset(image_ids, image_path)
    test_loader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=10, pin_memory=False)


    model = UNet16(num_classes=5, pretrained='vgg')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    print('load model weight')
    state = torch.load(model_weight)
    model.load_state_dict(state['model'])

    cudnn.benchmark = True

    attr_types = ['pigment_network', 'negative_network', 'streaks', 'milia_like_cyst', 'globules']
    with torch.no_grad():
        for img_id, test_image, W, H in test_loader:
            img_id = img_id[0]
            W = W[0].item()
            H = H[0].item()
            print(img_id, W, H, test_image.size())
            test_image = test_image.to(device)  # [N, 1, H, W]
            test_image = test_image.permute(0, 3, 1, 2)
            outputs, outputs_mask_ind1, outputs_mask_ind2 = model(test_image)
            test_prob = F.sigmoid(outputs)
            test_prob = test_prob.squeeze().data.cpu().numpy()
            for ind, attr in enumerate(attr_types):
                resize_mask = cv2.resize(test_prob[ind, :, :], (W, H), interpolation=cv2.INTER_CUBIC)
                #for cutoff in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                for cutoff in [0.3]:
                    if not os.path.exists('submission_%s'%(cutoff)): os.makedirs('submission_%s'%(cutoff))
                    test_mask = (resize_mask>cutoff).astype('int') *255.0
                    cv2.imwrite('submission_%s/'%(cutoff) + "ISIC_%s_attribute_%s.png" % (img_id.split('_')[1],attr), test_mask)
            #break

## cutoff validation jaccard
## 0.0 0.012
## 0.1 0.475
## 0.2 0.476
## 0.3 0.477
## 0.4 0.477
## 0.5 0.477
def main():
    model_weight = 'runs_three_losses_part2/debug/model.pt'
    image_path = '/media/eric/HDD1/1_Project_Raw_Data/23_ISIC_2018/0_Data/Task2/ISIC2018_Task1-2_Validation_Input/'
    #image_path = '/media/eric/SSD2/Project/11_ISCB2018/0_Data/Task2/valid_h5/'
    print(image_path)
    test_new_data(model_weight, image_path)



if __name__ == '__main__':
    # python train.py --image-path /media/eric/SSD2/Project/11_ISCB2018/0_Data/Task2/h5/
    # tensorboard --logdir runs --port 6008
    main()