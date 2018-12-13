#!/usr/bin/env python3
# -------------------------------------------------------------
# proj:    gan-stronomy
# file:    score.py
# authors: Mark Sabini, Zahra Abdullah, Darrith Phan
# desc:    Computes FID as a quantitative metric of image quality.
# refs:    https://github.com/mseitzer/pytorch-fid
# -------------------------------------------------------------
import torch
from torch.autograd import Variable
from dataset import GANstronomyDataset
import sys
import os
from model import Generator, Discriminator
from torch.utils.data import DataLoader, SequentialSampler
import util
import opts

def get_val_imgs(G, data):
    data.set_split_index(1) # Set to validation split
    data_loader = DataLoader(data, batch_size=len(data), shuffle=False, sampler=SequentialSampler(data))
    for ibatch, data_batch in enumerate(data_loader):
        with torch.no_grad():
            recipe_ids, recipe_embs, img_ids, imgs, classes, _, _ = data_batch
            batch_size, recipe_embs, imgs, = util.get_variables3(recipe_ids, recipe_embs, img_ids, imgs)
            z = torch.randn(batch_size, opts.LATENT_SIZE).to(opts.DEVICE)
            imgs_gen = G(z, recipe_embs)
            return imgs.detach(), imgs_gen.detach()

def get_test_imgs(data):
    data.set_split_index(2)
    data_loader = DataLoader(data, batch_size=len(data), shuffle=False, sampler=SequentialSampler(data))
    for ibatch, data_batch in enumerate(data_loader):
        recipe_ids, recipe_embs, img_ids, imgs, classes, _, _ = data_batch
        batch_size, recipe_embs, imgs, = util.get_variables3(recipe_ids, recipe_embs, img_ids, imgs)
        return imgs.detach()

def main():
    if len(sys.argv) < 3:
        print('Usage: python3 score.py [MODEL_PATH] [DATA_PATH]')
        exit()
    model_path = os.path.abspath(sys.argv[1])
    data_path = os.path.abspath(sys.argv[2])

    saved_model = torch.load(model_path)

    data = GANstronomyDataset(data_path, split=opts.TVT_SPLIT)
    
    G = Generator(opts.LATENT_SIZE, opts.EMBED_SIZE).to(opts.DEVICE)
    G.load_state_dict(saved_model['G_state_dict'])
    G.eval()

    val_imgs, val_imgs_gen = get_val_imgs(G, data)
    test_imgs = get_test_imgs(data)
    print('FID(test_real, val_real): %f' % util.get_fid(test_imgs, val_imgs))
    print('FID(test_real, val_fake): %f' % util.get_fid(test_imgs, val_imgs_gen))
   
if __name__ == '__main__':
    main()
