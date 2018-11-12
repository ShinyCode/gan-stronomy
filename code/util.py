import imp
import numpy as np
import pickle
from PIL import Image
import sys
import os
import json

IMAGE_SZ = 128
KEY_PATH = os.path.abspath('../../data/val_keys.pkl')
CLASS_PATH = os.path.abspath('../../data/classes1M.pkl')
RAW_IMG_PATH = os.path.abspath('../../data/val_raw')
RSZ_IMG_PATH = os.path.abspath('../../data/val_rsz2')
INGR_PATH = os.path.abspath('../../data/det_ingrs.json')

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
