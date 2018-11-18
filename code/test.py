import torch
from torch.autograd import Variable
from dataset import GANstronomyDataset
import sys
import os
from model import Generator, Discriminator
import opts
from torch.utils.data import DataLoader

def main():
    if len(sys.argv) < 5:
        print('Usage: python3 test.py [MODEL_PATH] [DATA_PATH] [SPLIT_INDEX] [OUT_PATH]')
        exit()
    model_path = os.path.abspath(sys.argv[1])
    data_path = os.path.abspath(sys.argv[2])
    split_index = int(sys.argv[3])
    out_path = os.path.abspath(sys.argv[3])

    saved_model = torch.load(model_path)

    data = GANstronomyDataset(data_path, split=opts.TVT_SPLIT)
    data.set_split_index(split_index)
    data_loader = DataLoader(data, batch_size=opts.BATCH_SIZE, shuffle=False)

    num_classes = data.num_classes()
    G = Generator(opts.EMBED_SIZE, num_classes).to(opts.DEVICE)
    G.load_state_dict(saved_model['G_state_dict'])
    G.eval()

    for ibatch, data_batch in enumerate(data_loader):
        with torch.no_grad():
            recipe_ids, recipe_embs, img_ids, imgs, classes = data_batch
            batch_size, recipe_embs, imgs, classes, classes_one_hot = None

    

if __name__ == '__main__':
    main()
