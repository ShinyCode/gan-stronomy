# -------------------------------------------------------------
# proj:    gan-stronomy
# file:    sample.py
# authors: Mark Sabini, Zahra Abdullah, Darrith Phan
# desc:    Executable to sample points from model given a recipe embedding id.
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

# [recipe_id]_[i].png
def save_results(out_path, recipe_id, img_gen, i):
    out_path = os.path.abspath(out_path)
    print('Saving results for i = %d to %s...' % (i, out_path))
    util.save_img(img_gen, out_path, '_'.join([recipe_id, '%d' % i]) + '.png')
            
def main():
    if len(sys.argv) < 7:
        print('Usage: python3 sample.py [MODEL_PATH] [DATA_PATH] [SPLIT_INDEX] [OUT_PATH] [RECIPE_ID] [NUM_SAMPLES]')
        exit()
    model_path = os.path.abspath(sys.argv[1])
    data_path = os.path.abspath(sys.argv[2])
    split_index = int(sys.argv[3])
    out_path = os.path.abspath(sys.argv[4])
    recipe_id = sys.argv[5]
    num_samples = int(sys.argv[6])
    
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
            
    z = torch.randn(num_samples, opts.LATENT_SIZE).to(opts.DEVICE)
    imgs_gen = G(z, embs.expand(num_samples, opts.EMBED_SIZE)).detach()
    for i in range(num_samples):
        img_gen = imgs_gen[i]
        save_results(out_path, recipe_id, img_gen, i)
    
if __name__ == '__main__':
    main()
