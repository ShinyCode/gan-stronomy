# -------------------------------------------------------------
# proj:    gan-stronomy
# file:    test.py
# authors: Mark Sabini, Zahra Abdullah, Darrith Phan
# desc:    Executable to run the model on images belonging to a split.
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

# [recipe_id]_[img_id]_real.png
# [recipe_id]_[img_id]_fake.png
# [recipe_id]_[img_id]_text.png
def save_results(all_ingrs, img, img_gen, img_id, recipe_id, out_path):
    out_path = os.path.abspath(out_path)
    print('Saving results for recipe %s to %s...' % (recipe_id, out_path))
    util.save_img(img, out_path, '_'.join([recipe_id, img_id, 'real']) + '.png')
    util.save_img(img_gen, out_path, '_'.join([recipe_id, img_id, 'fake']) + '.png')
    ingrs = util.get_valid_ingrs(all_ingrs, recipe_id)
    ingr_filename = '_'.join([recipe_id, img_id, 'text']) + '.txt'
    with open(os.path.join(out_path, ingr_filename), 'w') as f:
        for ingr in ingrs:
            f.write('%s\n' % ingr)
            
def main():
    if len(sys.argv) < 5:
        print('Usage: python3 test.py [MODEL_PATH] [DATA_PATH] [SPLIT_INDEX] [OUT_PATH]')
        exit()
    model_path = os.path.abspath(sys.argv[1])
    data_path = os.path.abspath(sys.argv[2])
    split_index = int(sys.argv[3])
    out_path = os.path.abspath(sys.argv[4])

    util.create_dir(out_path)
    saved_model = torch.load(model_path)

    data = GANstronomyDataset(data_path, split=opts.TVT_SPLIT)
    data.set_split_index(split_index)
    data_loader = DataLoader(data, batch_size=opts.BATCH_SIZE, shuffle=False, sampler=SequentialSampler(data))
    
    G = Generator(opts.LATENT_SIZE, opts.EMBED_SIZE).to(opts.DEVICE)
    G.load_state_dict(saved_model['G_state_dict'])
    G.eval()

    all_ingrs = util.load_ingredients()

    for ibatch, data_batch in enumerate(data_loader):
        with torch.no_grad():
            recipe_ids, recipe_embs, img_ids, imgs, classes, _, _ = data_batch
            batch_size, recipe_embs, imgs, = util.get_variables3(recipe_ids, recipe_embs, img_ids, imgs)
            z = torch.randn(batch_size, opts.LATENT_SIZE).to(opts.DEVICE)
            imgs_gen = G(z, recipe_embs)
            imgs, imgs_gen = imgs.detach(), imgs_gen.detach()
            for iexample in range(batch_size):
                save_results(all_ingrs, imgs[iexample], imgs_gen[iexample], img_ids[iexample], recipe_ids[iexample], out_path)

if __name__ == '__main__':
    main()
