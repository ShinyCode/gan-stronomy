#!/usr/bin/env python3
# -------------------------------------------------------------
# proj:    gan-stronomy
# file:    interp.py
# authors: Mark Sabini, Zahra Abdullah, Darrith Phan
# desc:    Executable to perform latent space interpolation.
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
from opts import LongTensor, FloatTensor
import numpy as np

# [recipe_id]_[a].png
def save_results(out_path, recipe_id, img_gen, a):
    out_path = os.path.abspath(out_path)
    print('Saving results for a = %.3f to %s...' % (a, out_path))
    util.save_img(img_gen, out_path, '_'.join([recipe_id, '%.3f' % a]) + '.png')
            
def main():
    if len(sys.argv) < 7:
        print('Usage: python3 interp.py [MODEL_PATH] [DATA_PATH] [SPLIT_INDEX] [OUT_PATH] [RECIPE_ID] [NUM_DIV]')
        exit()
    model_path = os.path.abspath(sys.argv[1])
    data_path = os.path.abspath(sys.argv[2])
    split_index = int(sys.argv[3])
    out_path = os.path.abspath(sys.argv[4])
    recipe_id = sys.argv[5]
    num_div = int(sys.argv[6])
    
    util.create_dir(out_path)
    saved_model = torch.load(model_path)

    data = GANstronomyDataset(data_path, split=opts.TVT_SPLIT)
    data.set_split_index(split_index)
    data_loader = DataLoader(data, batch_size=1, shuffle=False, sampler=SequentialSampler(data))

    num_classes = data.num_classes()
    G = Generator(opts.LATENT_SIZE, opts.EMBED_SIZE).to(opts.DEVICE)
    G.load_state_dict(saved_model['G_state_dict'])
    G.eval()

    embs = None

    for ibatch, data_batch in enumerate(data_loader):
        with torch.no_grad():
            recipe_ids, recipe_embs, img_ids, imgs, classes, _, _ = data_batch
            batch_size, recipe_embs, imgs  = util.get_variables3(recipe_ids, recipe_embs, img_ids, imgs)
            if recipe_ids[0] == recipe_id:
                embs = recipe_embs
                break

    assert embs is not None
    z1 = torch.randn(1, opts.LATENT_SIZE).to(opts.DEVICE)
    z2 = torch.randn(1, opts.LATENT_SIZE).to(opts.DEVICE)
    for a in np.linspace(0.0, 1.0, num_div + 1):
        a = torch.tensor(a, dtype=torch.float)
        a = Variable(a.type(FloatTensor)).to(opts.DEVICE)
        z = (1.0 - a) * z1 + a * z2
        img_gen = G(z, embs).detach()[0]
        save_results(out_path, recipe_id, img_gen, a)

if __name__ == '__main__':
    main()
