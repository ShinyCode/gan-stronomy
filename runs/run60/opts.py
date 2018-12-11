import torch
import os
import util

# General parameters
SAFETY_MODE = False

# Device parameters
CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if CUDA else 'cpu'
FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if CUDA else torch.LongTensor
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# Data parameters
EMBED_SIZE = 1024
LATENT_SIZE = 100
IMAGE_SIZE = 64
# TVT_SPLIT = [0.95, 0.025, 0.025]
# TVT_SPLIT = [49800, 100, 100]
# TVT_SPLIT = [9950, 50, 0]
TVT_SPLIT = [950, 50, 0]
TVT_SPLIT_LABELS = ['train', 'val', 'test']
DATASET_NAME = 'data1000'
DATA_PATH = os.path.abspath('../temp/%s/data.pkl' % DATASET_NAME)

# Training parameters
BATCH_SIZE = 64
ADAM_LR = 0.0002
ADAM_B = (0.5, 0.999)
NUM_EPOCHS = 61
LAMBDA = 10.0 # Weight of gradient penalty

# Model parameters
NGF = 64
NDF = 64
USE_CLASSES = True

# Run parameters
RUN_ID = 60
RUN_COMMENT = 'With latent space and gan loss'
RUN_PATH = os.path.abspath('../runs/run%d' % RUN_ID)
IMG_OUT_PATH = os.path.join(RUN_PATH, 'out')
# MODEL_PATH = '../runs/run11/models/model_run11_data100_100_0.pt'
MODEL_PATH = None # None means starting fresh
MODEL_OUT_PATH = os.path.join(RUN_PATH, 'models')
INTV_PRINT_LOSS = 1
INTV_SAVE_IMG = 1
INTV_SAVE_MODEL = 10
NUM_UPDATE_D = 3
