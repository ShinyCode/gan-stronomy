import torch
from torch.autograd import Variable
from dataset import GANstronomyDataset
import sys
import os
from model import Generator, Discriminator
import opts
from torch.utils.data import DataLoader
import util

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

    saved_model = torch.load(model_path)

    data = GANstronomyDataset(data_path, split=opts.TVT_SPLIT)
    data.set_split_index(split_index)
    data_loader = DataLoader(data, batch_size=opts.BATCH_SIZE, shuffle=False)

    num_classes = data.num_classes()
    G = Generator(opts.EMBED_SIZE, num_classes).to(opts.DEVICE)
    G.load_state_dict(saved_model['G_state_dict'])
    G.eval()

    all_ingrs = util.load_ingredients()

    for ibatch, data_batch in enumerate(data_loader):
        with torch.no_grad():
            recipe_ids, recipe_embs, img_ids, imgs, classes = data_batch
            batch_size, recipe_embs, imgs, classes, classes_one_hot = util.get_variables(recipe_ids, recipe_embs, img_ids, imgs, classes, num_classes)
            imgs_gen = G(recipe_embs, classes_one_hot)
            imgs, imgs_gen = imgs.detach(), imgs_gen.detach()
            for iexample in range(batch_size):
                save_results(all_ingrs, imgs[iexample], imgs_gen[iexample], img_ids[iexample], recipe_ids[iexample], out_path)

if __name__ == '__main__':
    main()
