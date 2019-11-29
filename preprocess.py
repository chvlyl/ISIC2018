import os
import numpy as np
import pandas as pd
import pickle
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import h5py
from joblib import Parallel, delayed



def load_image(ind,row,args):
    if ind % 50 == 0:
        print('processing %s' % ind)

    mask_path = args.mask_path
    image_path = args.image_path
    save_path = args.save_path

    #all_images = np.zeros(shape=(train_test_id.shape[0],512,512,3))
    #for ind, row in train_test_id.iterrows():
    img_id = row.ID
    labels  = row[['pigment_network','negative_network','streaks','milia_like_cyst','globules']].values
    
    ###############
    ### load image
    image_file = image_path + '%s.jpg' % img_id
    img = load_img(image_file, target_size=(512,512), color_mode="rgb")  # this is a PIL image
    img_np = img_to_array(img)
    ### why only 0-255 integers
    img_np = img_np.astype(np.uint8)
    hdf5_file = h5py.File(save_path + '%s.h5' % img_id, 'w')
    hdf5_file.create_dataset('img', data=img_np, dtype=np.uint8)
    hdf5_file.close()
    
    ################
    ### load masks
    attr_types = ['pigment_network', 'negative_network', 'streaks', 'milia_like_cyst', 'globules']
    masks = np.zeros(shape=(img_np.shape[0], img_np.shape[1], 5))
    for i, attr in enumerate(attr_types):
        mask_file = mask_path + '%s_attribute_%s.png' % (img_id, attr)
        m = load_img(mask_file, target_size=(512,512), color_mode="grayscale")  # this is a PIL image
        m_np = img_to_array(m)
        masks[:, :, i] = m_np[:, :, 0]
        m_np = m_np[:, :, 0, np.newaxis]
        m_np = (m_np / 255).astype('int8')
        hdf5_file = h5py.File(save_path + '%s_attribute_%s.h5' % (img_id, attr), 'w')
        hdf5_file.create_dataset('img', data=m_np, dtype=np.int8)
        hdf5_file.close()
    masks = (masks / 255).astype('int8')
    hdf5_file = h5py.File(save_path + '%s_attribute_all.h5' % (img_id), 'w')
    hdf5_file.create_dataset('img', data=masks, dtype=np.int8)
    hdf5_file.close()
    # print(img_np.shape,masks.shape)
    ##########################
    #masks = masks.astype('uint8')
    return None

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--train-test-split-file', type=str, default='./data/train_test_id.pickle', help='train test split file')
    arg('--image-path', type=str, default='./data/ISIC2018_Task1-2_Training_Input/', help='training image path')
    arg('--mask-path', type=str, default='./data/ISIC2018_Task2_Training_GroundTruth_v3/', help='ground truth mask path')
    arg('--save-path', type=str, default='./data/task2_h5/', help='output path')
    
    args = parser.parse_args()

    print('load %s'%args.train_test_split_file)
    with open(args.train_test_split_file, 'rb') as f:
        train_test_id = pickle.load(f)

    train_test_id['all'] = train_test_id[['pigment_network','negative_network','streaks','milia_like_cyst','globules']].sum(axis=1)
    print(train_test_id['all'].value_counts())
    # 4       7
    # 3     181
    # 0     514
    # 2     635
    # 1    1257

    ## save train_test_split.csv
    train_test_id.to_csv('./data/train_test_id.csv',index=False)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    results = Parallel(n_jobs=12)(delayed(load_image)(ind,row, args) for ind,row in train_test_id.iterrows())


if __name__ == '__main__':
    main()