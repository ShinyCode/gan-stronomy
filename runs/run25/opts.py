import torch
import os

# General parameters
SAFETY_MODE = False

# Device parameters
CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if CUDA else 'cpu'
FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if CUDA else torch.LongTensor

# Data parameters
EMBED_SIZE = 1024
IMAGE_SIZE = 64
# TVT_SPLIT = [0.95, 0.025, 0.025]
TVT_SPLIT = [95, 5, 0]
TVT_SPLIT_LABELS = ['train', 'val', 'test']
DATASET_NAME = 'data100'
DATA_PATH = os.path.abspath('../temp/%s/data.pkl' % DATASET_NAME)

# Training parameters
BATCH_SIZE = 16
ADAM_LR = 0.0002
ADAM_B = (0.5, 0.999)
NUM_EPOCHS = 201
ALPHA = 0.01 # ALPHA = 0 means all MSE loss

# Model parameters
NGF = 64
NDF = 64
USE_CLASSES = False

# Run parameters
RUN_ID = 25
RUN_PATH = os.path.abspath('../runs/run%d' % RUN_ID)
IMG_OUT_PATH = os.path.join(RUN_PATH, 'out')
# MODEL_PATH = '../runs/run11/models/model_run11_data100_100_0.pt'
MODEL_PATH = None # None means starting fresh
MODEL_OUT_PATH = os.path.join(RUN_PATH, 'models')
INTV_PRINT_LOSS = 1
INTV_SAVE_IMG = 50
INTV_SAVE_MODEL = 50
NUM_UPDATE_G = 1
NUM_UPDATE_D = 1
