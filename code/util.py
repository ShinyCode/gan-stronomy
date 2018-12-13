import imp
import numpy as np
import pickle
from PIL import Image
import sys
import os
import json
import lmdb
import word2vec
import random
import opts
from opts import FloatTensor, LongTensor
from torch.autograd import Variable
import time
import datetime
import torch
import fid_score

IMAGE_SZ = 64
DATA_ROOT = os.path.abspath('../../data')
KEY_PATH = os.path.join(DATA_ROOT, 'val_keys.pkl')
CLASS_PATH = os.path.join(DATA_ROOT, 'classes1M.pkl')
INGR_PATH = os.path.join(DATA_ROOT, 'det_ingrs.json')
LMDB_PATH = os.path.join(DATA_ROOT, 'val_lmdb/')
RAW_IMG_PATH = os.path.join(DATA_ROOT, 'val_raw')
RSZ_IMG_PATH = os.path.join(DATA_ROOT, 'val_rsz2')
IMG_ID_PATH = os.path.join(DATA_ROOT, 'img_ids.txt')
VOCAB_PATH = os.path.join(DATA_ROOT, 'vocab.bin')
IM2RECIPE_ROOT = os.path.abspath('../../im2recipe-Pytorch')
GEN_EMBEDDINGS_ROOT = os.path.join(IM2RECIPE_ROOT, 'gen_embeddings.py')
MODEL_PATH = os.path.join(DATA_ROOT, 'model_e500_v-8.950.pth.tar')

def get_time():
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

def reload():
    imp.reload(sys.modules[__name__])

def unpickle(filename, i=0):
    data = None
    with open(filename, 'rb') as f:
        for _ in range(i+1):
            data = pickle.load(f)
    return data

def unpickle2(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

def repickle(obj, out_path):
    with open(out_path, 'wb') as f:
        pickle.dump(obj, f, protocol=2)

def load_ids():
    return unpickle(KEY_PATH)

def load_classes():
    return unpickle(CLASS_PATH)

def load_class_names():
    return unpickle(CLASS_PATH, 1)

def load_ingredients():
    with open(INGR_PATH) as f:
        lines = ''.join([line.rstrip() for line in f])
    lines = '{"data": '+ lines + '}'
    lines_parsed = json.loads(lines)
    ingrs = {}
    for entry in lines_parsed['data']:
        ingrs[entry['id']] = entry
    return ingrs

def load_lmdb(lmdb_file=LMDB_PATH):
    lmdb_env = lmdb.open(lmdb_file)
    with lmdb_env.begin() as lmdb_txn:
        lmdb_cursor = lmdb_txn.cursor()
        lmdb_data = {}
        for key, value in lmdb_cursor:
            lmdb_data[key] = pickle.loads(value, encoding='latin1')
    return lmdb_data

def sample_ids(lmdb_data, N):
    return random.sample(list(lmdb_data.keys()), N)

def slice_lmdb(ids, lmdb_data):
    ids = set(ids)
    return {k:lmdb_data[k] for k in lmdb_data if k in ids}

def save_lmdb_data(lmdb_data, out_path):
    lmdb_env = lmdb.open(out_path, map_size=int(1e11))
    with lmdb_env.begin(write=True) as lmdb_txn:
        for key in lmdb_data:
            lmdb_txn.put(key, pickle.dumps(lmdb_data[key], protocol=2))
    print("Done saving lmdb_data to %s." % out_path)

# Returns a dict mapping recipe IDs to img IDs
def map_recipe_id_to_img_id(lmdb_data):
    mapping = {}
    for key, value in lmdb_data.items():
        img_ids = []
        for img in value['imgs']:
            img_ids.append(img['id'].split('.')[0])
        mapping[key] = img_ids
    return mapping

# Returns numpy array of size (IMAGE_SZ, IMAGE_SZ, 3)
def resize_crop_img(img_filename):
    img = Image.open(img_filename).convert('RGB')
    width, height = img.size
    ratio = float(IMAGE_SZ) / min(width, height)
    if width <= height: # skinny
        img_rsz = img.resize((IMAGE_SZ, int(ratio * height)), Image.ANTIALIAS)
        y_start = int((img_rsz.height - IMAGE_SZ) / 2)
        img_crop = img_rsz.crop(box=(0, y_start, IMAGE_SZ, y_start + IMAGE_SZ))
    else: # wide
        img_rsz = img.resize((int(ratio * width), IMAGE_SZ), Image.ANTIALIAS)
        x_start = int((img_rsz.width - IMAGE_SZ) / 2)
        img_crop = img_rsz.crop(box=(x_start, 0, x_start + IMAGE_SZ, IMAGE_SZ))
    assert img_crop.size == (IMAGE_SZ, IMAGE_SZ)
    return img_crop

def get_img_path(id, raw_img_path):
    filename = id + '.jpg'
    return os.path.join(raw_img_path, id[0], id[1], id[2], id[3], filename)

def resize_crop_imgs(in_path, out_path, sample=100, verbose=False):
    i = 0
    for dirpath, dirnames, filenames in os.walk(in_path):
        for in_filename in filenames:
            if i % sample == 0:
                img_crop = resize_crop_img(os.path.join(dirpath, in_filename))
                id = in_filename.split('.')[0]
                out_filename = id + '.png'
                if verbose:
                    print('Saving %s...' % out_filename)
                img_crop.save(os.path.join(out_path, out_filename), format='PNG')
            i += 1

def get_img_ids(in_path, out_path):
    ids = []
    for dirpath, dirnames, filenames in os.walk(in_path):
        for in_filename in filenames:
            id = in_filename.split('.')[0]
            ids.append(id)

    with open(out_path, 'w') as f:
        for id in ids:
            f.write(id + '\n')

def get_vocab(out_path):
    model = word2vec.load(VOCAB_PATH)
    vocab = model.vocab
    f = open(out_path, 'w')
    f.write("\n".join(vocab))
    f.close()

def create_dir(dir_path):
    try:
        os.stat(dir_path)
    except:
        os.mkdir(dir_path)

def get_variables(recipe_ids, recipe_embs, img_ids, imgs, classes, num_classes):
    # Set up Variables
    batch_size = imgs.shape[0]
    recipe_embs = Variable(recipe_embs.type(FloatTensor)).to(opts.DEVICE)
    imgs = Variable(imgs.type(FloatTensor)).to(opts.DEVICE)
    classes = Variable(classes.type(LongTensor)).to(opts.DEVICE)
    classes_one_hot = Variable(FloatTensor(batch_size, num_classes).zero_().scatter_(1, classes.view(-1, 1), 1)).to(opts.DEVICE)
    return batch_size, recipe_embs, imgs, classes, classes_one_hot

def get_variables2(noisy_real, noisy_fake):
    noisy_real = Variable(noisy_real.type(FloatTensor), requires_grad=False).to(opts.DEVICE)
    noisy_fake = Variable(noisy_fake.type(FloatTensor), requires_grad=False).to(opts.DEVICE)
    return noisy_real[:, None], noisy_fake[:, None]

def get_variables3(recipe_ids, recipe_embs, img_ids, imgs):
    batch_size = imgs.shape[0]
    recipe_embs = Variable(recipe_embs.type(FloatTensor)).to(opts.DEVICE)
    imgs = Variable(imgs.type(FloatTensor)).to(opts.DEVICE)
    return batch_size, recipe_embs, imgs
    
# Assumes image values are in [-1, 1]
def save_img(img, out_path, filename):
    out_path = os.path.abspath(out_path)
    img_scale = (img + 1.0) / 2.0
    img = np.transpose(np.array(255.0 * img_scale, dtype=np.uint8), (1, 2, 0))
    img_png = Image.fromarray(img, mode='RGB')
    img_png.save(os.path.join(out_path, filename), format='PNG')

def get_valid_ingrs(ingrs, recipe_id):
    ret = set()
    for ingr, valid in zip(ingrs[recipe_id]['ingredients'], ingrs[recipe_id]['valid']):
        if valid:
            ret.add(ingr['text'])
    return ret

def get_fid(imgs1, imgs2):
    return fid_score.calculate_fid_given_arrays(imgs1, imgs2)        
    
    
