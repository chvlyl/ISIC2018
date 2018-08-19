import numpy as np
import pandas as pd
import pickle
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#from sklearn.model_selection import train_test_split
import h5py
from joblib import Parallel, delayed



def load_image(ind,row):
    if ind % 50 == 0:
        print(ind)
    mask_path = '/media/eric/HDD1/1_Project_Raw_Data/23_ISIC_2018/0_Data/Task2/ISIC2018_Task2_Training_GroundTruth_v3/'
    image_path = '/media/eric/HDD1/1_Project_Raw_Data/23_ISIC_2018/0_Data/Task2/ISIC2018_Task1-2_Training_Input/'

    save_path = '/media/eric/SSD2/Project/11_ISCB2018/0_Data/Task2/h5/'

    #all_images = np.zeros(shape=(train_test_id.shape[0],512,512,3))
    #for ind, row in train_test_id.iterrows():
    img_id = row.ID
    labels  = row[['pigment_network','negative_network','streaks','milia_like_cyst','globules']].values
    ###############
    ### load image
    image_file = image_path + '%s.jpg' % img_id
    img = load_img(image_file, target_size=(512,512), grayscale=False)  # this is a PIL image
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
        m = load_img(mask_file, target_size=(512,512), grayscale=True)  # this is a PIL image
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


with open('/media/eric/SSD2/Project/11_ISCB2018/2_Analysis/20180516_classification_segmentation/train_test_id.pickle', 'rb') as f:
    train_test_id = pickle.load(f)

train_test_id['all'] = train_test_id[['pigment_network','negative_network','streaks','milia_like_cyst','globules']].sum(axis=1)
train_test_id['all'].value_counts()
# 4       7
# 3     181
# 0     514
# 2     635
# 1    1257

results = Parallel(n_jobs=12)(delayed(load_image)(ind,row) for ind,row in train_test_id.iterrows())


####################################
####################################
#### test code
import h5py
mask_path = '/media/eric/HDD1/1_Project_Raw_Data/23_ISIC_2018/0_Data/Task2/ISIC2018_Task2_Training_GroundTruth_v3/'
image_path = '/media/eric/HDD1/1_Project_Raw_Data/23_ISIC_2018/0_Data/Task2/ISIC2018_Task1-2_Training_Input/'
img_id = 'ISIC_0000164'
image_file = image_path + '%s.jpg' % img_id
img = load_img(image_file,  grayscale=False)  # this is a PIL image
img_np = img_to_array(img)
img_np = img_np.astype(np.uint8)


attr_types = ['pigment_network', 'negative_network', 'streaks', 'milia_like_cyst', 'globules']
for i, attr in enumerate(attr_types):
    mask_file = mask_path + '%s_attribute_%s.png' % (img_id, attr)
    m = load_img(mask_file, grayscale=True)  # this is a PIL image
    m_np = img_to_array(m)[:, :, 0]
    m_np = (m_np / 255).astype('int8')
    # hdf5_file = h5py.File(
    #     '/media/eric/SSD2/Project/11_ISCB2018/2_Analysis/20180618_pytorch_unet_classfication_segmentation/test.h5', 'w')
    # hdf5_file.create_dataset('img', data=m_np, dtype=np.int8)
    # hdf5_file.close()
    break

## cumulative sum > 0,


# # open a hdf5 file and create earrays
# hdf5_file = h5py.File('/media/eric/SSD2/Project/11_ISCB2018/2_Analysis/20180618_pytorch_unet_classfication_segmentation/test.h5', 'w')
# #hdf5_file.create_dataset("img", test_shape, np.int8)
# hdf5_file.create_dataset('img', data=img_np, dtype=np.uint8)
# hdf5_file.close()
#
# attr_types = ['pigment_network', 'negative_network', 'streaks', 'milia_like_cyst', 'globules']
# for i, attr in enumerate(attr_types):
#     mask_file = mask_path + '%s_attribute_%s.png' % (img_id, attr)
#     m = load_img(mask_file, target_size=(512,512), grayscale=True)  # this is a PIL image
#     m_np = img_to_array(m)[:, :, 0]
#     m_np = (m_np / 255).astype('int8')
#     hdf5_file = h5py.File(
#         '/media/eric/SSD2/Project/11_ISCB2018/2_Analysis/20180618_pytorch_unet_classfication_segmentation/test.h5', 'w')
#     hdf5_file.create_dataset('img', data=m_np, dtype=np.int8)
#     hdf5_file.close()
#     break
