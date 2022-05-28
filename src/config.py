import torch
import os


class Config:
    # datasetについて
    DATA_DIR = '../input'
    compe_name = 'uw-madison-gi-tract-image-segmentation'
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    TRAIN_CSV = os.path.join(TRAIN_DIR, 'train.csv')
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    SS_CSV = os.path.join(TEST_DIR, 'sample_submission.csv')

    seed = 101
    debug = False  # set debug=False for Full Training

    exp_name = 'Baselinev2'
    comment = 'LeViT-unet-224x224-frozenEnc'
    model_name = 'Unet'
    backbone = 'LeVit-384'
    train_bs = 64
    valid_bs = train_bs*2
    img_size = [224, 224]
    epochs = 30
    lr = 2e-3
    scheduler = 'CosineAnnealingLR'
    min_lr = 1e-6
    T_max = int(30000/train_bs*epochs)+50
    T_0 = 31
    warmup_epochs = 0
    wd = 1e-6
    n_accumulate = max(1, 32//train_bs)
    n_fold = 5
    num_classes = 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
