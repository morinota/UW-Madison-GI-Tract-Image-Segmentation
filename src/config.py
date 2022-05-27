
import kaggle
import os

class Config:
    # datasetについて
    DATA_DIR = '../input'
    compe_name = 'uw-madison-gi-tract-image-segmentation'
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    TRAIN_CSV = os.path.join(TRAIN_DIR, 'train.csv')
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    SS_CSV = os.path.join(TEST_DIR, 'sample_submission.csv')