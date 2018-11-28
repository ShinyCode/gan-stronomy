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

# [recipe_id1]_[recipe_id2]_[a].png
def save_results(out_path, recipe_id1, recipe_id2, img_gen, a):
    out_path = os.path.abspath(out_path)
    print('Saving results for a = %.3f to %s...' % (a, out_path))
    util.save_img(img_gen, out_path, '_'.join([recipe_id1, recipe_id2, '%.3f' % a]) + '.png')
            
def main():
    if len(sys.argv) < 8:
        print('Usage: python3 interp.py [MODEL_PATH] [DATA_PATH] [SPLIT_INDEX] [OUT_PATH] [RECIPE_ID1] [RECIPE_ID2] [NUM_DIV]')
        exit()
    model_path = os.path.abspath(sys.argv[1])
    data_path = os.path.abspath(sys.argv[2])
    split_index = int(sys.argv[3])
    out_path = os.path.abspath(sys.argv[4])
    recipe_id1 = sys.argv[5]
    recipe_id2 = sys.argv[6]
    num_div = int(sys.argv[7])
    
    util.create_dir(out_path)
    saved_model = torch.load(model_path)

    data = GANstronomyDataset(data_path, split=opts.TVT_SPLIT)
    data.set_split_index(split_index)
    data_loader = DataLoader(data, batch_size=1, shuffle=False, sampler=SequentialSampler(data))

    num_classes = data.num_classes()
    G = Generator(opts.EMBED_SIZE, num_classes).to(opts.DEVICE)
    G.load_state_dict(saved_model['G_state_dict'])
    G.eval()

    recipe_id1_embs, recipe_id1_classes_one_hot = None, None
    recipe_id2_embs, recipe_id2_classes_one_hot = None, None
    
    for ibatch, data_batch in enumerate(data_loader):
        with torch.no_grad():
            recipe_ids, recipe_embs, img_ids, imgs, classes, _, _ = data_batch
            batch_size, recipe_embs, imgs, classes, classes_one_hot = util.get_variables(recipe_ids, recipe_embs, img_ids, imgs, classes, num_classes)
            if recipe_ids[0] == recipe_id1:
                recipe_id1_embs = recipe_embs
                recipe_id1_classes_one_hot = classes_one_hot
            elif recipe_ids[0] == recipe_id2:
                recipe_id2_embs = recipe_embs
                recipe_id2_classes_one_hot = classes_one_hot

    assert recipe_id1_embs is not None
    assert recipe_id2_embs is not None
    for a in np.linspace(0.0, 1.0, num_div + 1):
        a = torch.tensor(a, dtype=torch.float)
        a = Variable(a.type(FloatTensor)).to(opts.DEVICE)
        embs = (1.0 - a) * recipe_id1_embs + a * recipe_id2_embs
        classes_one_hot = (1.0 - a) * recipe_id1_classes_one_hot + a * recipe_id2_classes_one_hot
        img_gen = G(embs, classes_one_hot).detach()[0]
        save_results(out_path, recipe_id1, recipe_id2, img_gen, a)

if __name__ == '__main__':
    main()
