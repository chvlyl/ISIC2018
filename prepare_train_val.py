import pickle
import pandas as pd

def get_split(fold=None):
    with open('/media/eric/SSD2/Project/11_ISCB2018/2_Analysis/20180516_classification_segmentation/train_test_id.pickle','rb') as f:
        train_test_id = pickle.load(f)

        train_test_id['total'] = train_test_id[['pigment_network',
                                      'negative_network',
                                      'streaks',
                                      'milia_like_cyst',
                                      'globules']].sum(axis=1)
        valid = train_test_id[train_test_id.Split != 'train'].copy()
        valid['Split'] = 'train'
        train_test_id = pd.concat([train_test_id, valid], axis=0)
    return train_test_id