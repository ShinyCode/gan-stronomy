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

# [recipe_id]_[img_id]_from_[old_class]_to_[new_class].png
def save_results(img_gen, img_id, recipe_id, out_path, old_class, new_class):
    out_path = os.path.abspath(out_path)
    print('Saving results for recipe %s to %s...' % (recipe_id, out_path))
    util.save_img(img_gen, out_path, '_'.join([recipe_id, img_id, 'from', str(old_class), 'to', str(new_class)]) + '.png')
            
def main():
    if len(sys.argv) < 7:
        print('Usage: python3 swap_classes.py [MODEL_PATH] [DATA_PATH] [SPLIT_INDEX] [OUT_PATH] [TARGET_MAPPED_CLASS] [TARGET_RECIPE_ID]')
        exit()
    model_path = os.path.abspath(sys.argv[1])
    data_path = os.path.abspath(sys.argv[2])
    split_index = int(sys.argv[3])
    out_path = os.path.abspath(sys.argv[4])
    target_class = int(sys.argv[5])
    target_recipe_id = sys.argv[6]
    
    util.create_dir(out_path)
    saved_model = torch.load(model_path)

    data = GANstronomyDataset(data_path, split=opts.TVT_SPLIT)
    data.set_split_index(split_index)
    data_loader = DataLoader(data, batch_size=opts.BATCH_SIZE, shuffle=False, sampler=SequentialSampler(data))

    num_classes = data.num_classes()
    assert target_class >= 0 and target_class < num_classes
    G = Generator(opts.EMBED_SIZE, num_classes).to(opts.DEVICE)
    G.load_state_dict(saved_model['G_state_dict'])
    G.eval()

    for ibatch, data_batch in enumerate(data_loader):
        with torch.no_grad():
            recipe_ids, recipe_embs, img_ids, imgs, classes, _, _ = data_batch
            batch_size, recipe_embs, imgs = util.get_variables3(recipe_ids, recipe_embs, img_ids, imgs)
            for iclass in range(target_class):
                target_classes = torch.empty(batch_size, 1, dtype=torch.long).fill_(iclass)
                target_classes = Variable(target_classes.type(LongTensor)).to(opts.DEVICE)
                target_class_one_hot = Variable(FloatTensor(batch_size, num_classes).zero_().scatter_(1, target_classes.view(-1, 1), 1)).to(opts.DEVICE)
                imgs_gen = G(recipe_embs, target_class_one_hot)
                imgs_gen = imgs_gen.detach()
                for iexample in range(batch_size):
                    if recipe_ids[iexample] != target_recipe_id:
                        continue
                    save_results(imgs_gen[iexample], img_ids[iexample], recipe_ids[iexample], out_path, int(classes[iexample]), iclass)

if __name__ == '__main__':
    main()
