import pickle
import pandas as pd

def get_split(fold=None):
    # folds = {0: [1, 3],
    #          1: [2, 5],
    #          2: [4, 8],
    #          3: [6, 7]}
    #
    # train_path = data_path / 'cropped_train'
    #
    # train_file_names = []
    # val_file_names = []
    #
    # for instrument_id in range(1, 9):
    #     if instrument_id in folds[fold]:
    #         val_file_names += list((train_path / ('instrument_dataset_' + str(instrument_id)) / 'images').glob('*'))
    #     else:
    #         train_file_names += list((train_path / ('instrument_dataset_' + str(instrument_id)) / 'images').glob('*'))
    #
    # return train_file_names, val_file_names

    with open('/media/eric/SSD2/Project/11_ISCB2018/2_Analysis/20180516_classification_segmentation/train_test_id.pickle','rb') as f:
        train_test_id = pickle.load(f)

        train_test_id['total'] = train_test_id[['pigment_network',
                                      'negative_network',
                                      'streaks',
                                      'milia_like_cyst',
                                      'globules']].sum(axis=1)
        #train_test_id = train_test_id[~((train_test_id.Split == 'train') & (train_test_id.total <= 2))]
        valid = train_test_id[train_test_id.Split != 'train'].copy()
        valid['Split'] = 'train'
        train_test_id = pd.concat([train_test_id, valid], axis=0)
    return train_test_id