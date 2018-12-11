import torch
from torch.autograd import Variable
from dataset import GANstronomyDataset
import sys
import os
from model import Generator, Discriminator
from torch.utils.data import DataLoader, SequentialSampler
import util
import opts
import inception_score as score

def main():
    if len(sys.argv) < 4:
        print('Usage: python3 score.py [MODEL_PATH] [DATA_PATH] [SPLIT_INDEX]')
        exit()
    model_path = os.path.abspath(sys.argv[1])
    data_path = os.path.abspath(sys.argv[2])
    split_index = int(sys.argv[3])
    out_path = os.path.abspath(sys.argv[4])

    util.create_dir(out_path)

    data = GANstronomyDataset(data_path, split=opts.TVT_SPLIT)
    data.set_split_index(split_index)
    data_loader = DataLoader(data, batch_size=len(data), shuffle=False, sampler=SequentialSampler(data))

    all_ingrs = util.load_ingredients()
    saved_model = torch.load(model_path) #Keep Changing Model Path
    G = Generator(opts.LATENT_SIZE, opts.EMBED_SIZE).to(opts.DEVICE)
    G.load_state_dict(saved_model['G_state_dict'])
    G.eval()

    score_list = []
    for ibatch, data_batch in enumerate(data_loader):
        with torch.no_grad():
            recipe_ids, recipe_embs, img_ids, imgs, classes, _, _ = data_batch
            batch_size, recipe_embs, imgs, = util.get_variables3(recipe_ids, recipe_embs, img_ids, imgs)
            z = torch.randn(batch_size, opts.LATENT_SIZE).to(opts.DEVICE)
            imgs_gen = G(z, recipe_embs)
            imgs, imgs_gen = imgs_gen.detach()
            fake_score_mean, fake_score_std = score.inception_score(imgs_gen, batch_size=len(data.size))
            real_score_mean, real_score_std = score.inception_score(imgs, batch_size=len(data.size))
            score_list.append((fake_score_mean, fake_score_std, real_score_mean, real_score_std))

    print(score_list)

if __name__ == '__main__':
    main()
