import torch
from torch.autograd import Variable
from dataset import GANstronomyDataset
import sys
import os
from model import Generator, Discriminator
from torch.utils.data import DataLoader, SequentialSampler
import util
import opts
            
def main():
    if len(sys.argv) < 4:
        print('Usage: python3 score.py [MODEL_PATH] [DATA_PATH] [SPLIT_INDEX]')
        exit()
    model_path = os.path.abspath(sys.argv[1])
    data_path = os.path.abspath(sys.argv[2])
    split_index = int(sys.argv[3])

    saved_model = torch.load(model_path)

    data = GANstronomyDataset(data_path, split=opts.TVT_SPLIT)
    data.set_split_index(split_index)
    data_loader = DataLoader(data, batch_size=len(data), shuffle=False, sampler=SequentialSampler(data))
    
    G = Generator(opts.LATENT_SIZE, opts.EMBED_SIZE).to(opts.DEVICE)
    G.load_state_dict(saved_model['G_state_dict'])
    G.eval()

    all_ingrs = util.load_ingredients()
    
    for ibatch, data_batch in enumerate(data_loader):
        with torch.no_grad():
            recipe_ids, recipe_embs, img_ids, imgs, classes, _, _ = data_batch
            batch_size, recipe_embs, imgs, = util.get_variables3(recipe_ids, recipe_embs, img_ids, imgs)
            print(util.get_fid(G, recipe_embs, imgs))

if __name__ == '__main__':
    main()
