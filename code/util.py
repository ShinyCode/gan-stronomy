# http://im2recipe.csail.mit.edu/im2recipe-journal.pdf
import imp
import numpy as np
import pickle
from PIL import Image
import sys
import os
import json
import lmdb

IMAGE_SZ = 128
KEY_PATH = os.path.abspath('../../data/val_keys.pkl') # From val.tar
CLASS_PATH = os.path.abspath('../../data/classes1M.pkl')
INGR_PATH = os.path.abspath('../../data/det_ingrs.json')
LMDB_PATH = os.path.abspath('../../data/val_lmdb/')
RAW_IMG_PATH = os.path.abspath('../../data/val_raw')
RSZ_IMG_PATH = os.path.abspath('../../data/val_rsz2')
IMG_ID_PATH = os.path.abspath('../../data/img_ids.txt')


def reload():
    imp.reload(sys.modules[__name__])

def unpickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def load_ids():
    return unpickle(KEY_PATH)

def load_classes():
    return unpickle(CLASS_PATH)

def load_ingredients():
    with open(INGR_PATH) as f:
        lines = ''.join([line.rstrip() for line in f])
    lines = '{"data": '+ lines + '}'
    lines_parsed = json.loads(lines)
    ingrs = {}
    for entry in lines_parsed['data']:
        ingrs[entry['id']] = entry
    return ingrs

'''
key: '012b10e9b5'
value: {'ingrs': array([1089,   35,   40,  364, 1067, 8620,   53, 2614,   87,  373,    1,
          0,    0,    0,    0,    0,    0,    0,    0,    0], dtype=uint16), 'imgs': [{'url': 'http://tastykitchen.com/recipes/wp-content/uploads/sites/2/2015/01/IMG_4890-410x273.jpg', 'id': '07ab86480b.jpg'}], 'classes': 205, 'intrs': array([[ 5.1401235e-04, -3.3809550e-03,  4.7120253e-08, ...,
        -7.9159543e-02,  2.0456385e-02, -1.4754387e-06],
       [-1.2805461e-03, -7.5157797e-05,  8.2095166e-06, ...,
        -1.2427612e-01,  5.7611704e-02, -1.7387359e-06],
       [ 6.0865399e-03, -4.2897620e-05,  1.6133923e-08, ...,
        -1.7309278e-01,  9.2510087e-03, -3.7411391e-07],
       ...,
       [-4.1227862e-03, -3.4603923e-05,  6.6228072e-09, ...,
        -1.5870290e-01,  3.7285931e-02, -3.4891771e-06],
       [ 2.1693237e-02, -1.9400386e-06,  1.7962537e-07, ...,
        -1.9604881e-01,  1.1945767e-01, -2.3149775e-07],
       [ 6.5326065e-02, -2.1252763e-08,  3.6092853e-08, ...,
        -1.4798085e-01,  7.7384055e-02,  4.0524910e-11]], dtype=float32)}
'''
def load_lmdb():
    lmdb_file = LMDB_PATH
    lmdb_env = lmdb.open(lmdb_file)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    lmdb_data = {}
    for key, value in lmdb_cursor:
        lmdb_data[key] = pickle.loads(value, encoding='latin1')
    return lmdb_data

def map_id_to_imgs(lmdb_data):
    mapping = {}
    for key, value in lmdb_data:
        img_ids = []
        for img in value['imgs']:
            pass


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
