import torch
import os

# General parameters
SAFETY_MODE = True

# Device parameters
CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if CUDA else 'cpu'
FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if CUDA else torch.LongTensor

# Data parameters
EMBED_SIZE = 1024
IMAGE_SIZE = 64
# TVT_SPLIT = [0.95, 0.025, 0.025]
TVT_SPLIT = [1, 0, 0]
TVT_SPLIT_LABELS = ['train', 'val', 'test']
DATASET_NAME = 'data1'
DATA_PATH = os.path.abspath('../temp/%s/data.pkl' % DATASET_NAME)

# Training parameters
BATCH_SIZE = 1
ADAM_LR = 0.001
ADAM_B = (0.9, 0.999)
NUM_EPOCHS = 301
ALPHA = 0.0004

# Run parameters
RUN_ID = 10
RUN_PATH = os.path.abspath('../runs/run%d' % RUN_ID)
IMG_OUT_PATH = os.path.join(RUN_PATH, 'out')
MODEL_PATH = None # None means starting fresh
MODEL_OUT_PATH = os.path.join(RUN_PATH, 'models')
INTV_PRINT_LOSS = 50
INTV_SAVE_IMG = 50
