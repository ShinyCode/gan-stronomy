# Based loosely off https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
import util
import torch
from torch.autograd import Variable
import torch.optim
import torch.nn
import torch.utils.data
from dataset import GANstronomyDataset
import os
from model import Generator, Discriminator
from PIL import Image
import numpy as np
import opts
from opts import FloatTensor, LongTensor
import shutil

BCELoss = torch.nn.BCELoss()
MSELoss = torch.nn.MSELoss()

def wasserstein_loss(G, D, imgs_real, recipe_embs):
    batch_size = imgs_real.shape[0]
    z = torch.randn(batch_size, opts.LATENT_SIZE).to(opts.DEVICE)
    a = torch.rand(batch_size, 1, 1, 1).to(opts.DEVICE)
    imgs_r = a * G(z, recipe_embs) + (1.0 - a) * imgs_real
    dd = torch.autograd.grad(torch.sum(D(imgs_r, recipe_embs)), imgs_r, create_graph=True)[0]
    dd_norm = torch.sqrt(torch.sum(dd ** 2, dim=(1, 2, 3)))
    gp = torch.mean((dd_norm - 1.0) ** 2)
    D_loss = torch.mean(D(G(z, recipe_embs), recipe_embs)) - torch.mean(D(imgs_real, recipe_embs)) + opts.LAMBDA * gp
    G_loss = -torch.mean(D(G(z, recipe_embs), recipe_embs))
    return D_loss, G_loss

def gan_loss(G, D, imgs, recipe_embs, noisy_real, noisy_fake):
    batch_size = imgs.shape[0]
    z = torch.randn(batch_size, opts.LATENT_SIZE).to(opts.DEVICE)
    imgs_gen = G(z, recipe_embs)
    fake_probs = D(imgs_gen.detach(), recipe_embs)
    real_probs = D(imgs, recipe_embs)
    D_loss = BCELoss(fake_probs, noisy_fake) + BCELoss(real_probs, noisy_real)
    all_real = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False).to(opts.DEVICE)
    G_loss = BCELoss(fake_probs, all_real)
    return D_loss, G_loss

def get_img_gen(data, split_index, G, iepoch, out_path):
    old_split_index = data.split_index
    data.set_split_index(split_index)
    data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
    data_batch = next(iter(data_loader))
    with torch.no_grad():
        recipe_ids, recipe_embs, img_ids, imgs, classes, _, _ = data_batch
        batch_size, recipe_embs, imgs = util.get_variables3(recipe_ids, recipe_embs, img_ids, imgs)
        z = torch.randn(batch_size, opts.LATENT_SIZE).to(opts.DEVICE)
        imgs_gen = G(z, recipe_embs)
        save_img(imgs_gen[0], iepoch, out_path, split_index, recipe_ids[0], img_ids[0])
    data.set_split_index(old_split_index)

# img_gen is [3, 64, 64]
def save_img(img_gen, iepoch, out_path, split_index, recipe_id, img_id):
    filename = '_'.join([opts.TVT_SPLIT_LABELS[split_index], str(iepoch), recipe_id, img_id]) + '.png'
    util.save_img(img_gen, out_path, filename)

def print_loss(G_loss, D_loss, iepoch):
    print("[%s] Epoch: %d\tG: %f\tD: %f" % (util.get_time(), iepoch, G_loss, D_loss))

def save_model(G, G_optimizer, D, D_optimizer, iepoch, out_path):
    filename = '_'.join(['model', 'run%d' % opts.RUN_ID, opts.DATASET_NAME, str(iepoch)]) + '.pt'
    out_path = os.path.abspath(out_path)
    torch.save({
            'iepoch': iepoch,
            'G_state_dict': G.state_dict(),
            'G_optimizer_state_dict': G_optimizer.state_dict(),
            'D_state_dict': D.state_dict(),
            'D_optimizer_state_dict': D_optimizer.state_dict()
            }, os.path.join(out_path, filename))

def load_state_dicts(model_path, G, G_optimizer, D, D_optimizer):
    model_path = os.path.abspath(model_path)
    saved_model = torch.load(model_path)
    G.load_state_dict(saved_model['G_state_dict'])
    G_optimizer.load_state_dict(saved_model['G_optimizer_state_dict'])
    D.load_state_dict(saved_model['D_state_dict'])
    D_optimizer.load_state_dict(saved_model['D_optimizer_state_dict'])
    start_iepoch = saved_model['iepoch']
    start_ibatch = 1
    return start_iepoch, start_ibatch
    
def main():
    # Load the data
    data = GANstronomyDataset(opts.DATA_PATH, split=opts.TVT_SPLIT)
    data.set_split_index(0)
    data_loader = torch.utils.data.DataLoader(data,
                                              batch_size=opts.BATCH_SIZE,
                                              shuffle=True)

    # Make the output directory
    util.create_dir(opts.RUN_PATH)
    util.create_dir(opts.IMG_OUT_PATH)
    util.create_dir(opts.MODEL_OUT_PATH)

    # Copy opts.py and model.py to opts.RUN_PATH as a record
    shutil.copy2('opts.py', opts.RUN_PATH)
    shutil.copy2('model.py', opts.RUN_PATH)
    shutil.copy2('train.py', opts.RUN_PATH)
    
    # Instantiate the models
    G = Generator(opts.LATENT_SIZE, opts.EMBED_SIZE).to(opts.DEVICE)
    G_optimizer = torch.optim.Adam(G.parameters(), lr=opts.ADAM_LR, betas=opts.ADAM_B)

    D = Discriminator(opts.EMBED_SIZE).to(opts.DEVICE)
    D_optimizer = torch.optim.Adam(D.parameters(), lr=opts.ADAM_LR, betas=opts.ADAM_B)

    if opts.MODEL_PATH is None:
        start_iepoch, start_ibatch = 0, 0
    else:
        print('Attempting to resume training using model in %s...' % opts.MODEL_PATH)
        start_iepoch, start_ibatch = load_state_dicts(opts.MODEL_PATH, G, G_optimizer, D, D_optimizer)
    
    for iepoch in range(opts.NUM_EPOCHS):
        for ibatch, data_batch in enumerate(data_loader):
            # To try to resume training, just continue if iepoch and ibatch are less than their starts
            if iepoch < start_iepoch or (iepoch == start_iepoch and ibatch < start_ibatch):
                if iepoch % opts.INTV_PRINT_LOSS == 0 and not ibatch:
                    print('Skipping epoch %d...' % iepoch)
                continue
            
            recipe_ids, recipe_embs, img_ids, imgs, classes, noisy_real, noisy_fake = data_batch
            noisy_real, noisy_fake = util.get_variables2(noisy_real, noisy_fake)

            # Make sure we're not training on validation or test data!
            if opts.SAFETY_MODE:
                for recipe_id in recipe_ids:
                    assert data.get_recipe_split_index(recipe_id) == 0

            batch_size, recipe_embs, imgs = util.get_variables3(recipe_ids, recipe_embs, img_ids, imgs)

            # Adversarial ground truths
            all_real = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False).to(opts.DEVICE)
            # all_fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False).to(opts.DEVICE)

            z = torch.randn(batch_size, opts.LATENT_SIZE).to(opts.DEVICE)
            # D_loss, G_loss = wasserstein_loss(G, D, imgs, recipe_embs)
            
            # Train Discriminator
            z = torch.randn(batch_size, opts.LATENT_SIZE).to(opts.DEVICE)
            imgs_gen = G(z, recipe_embs)
            for _ in range(opts.NUM_UPDATE_D):
                D_optimizer.zero_grad()
                fake_probs = D(imgs_gen.detach(), recipe_embs)
                real_probs = D(imgs, recipe_embs)
                D_loss = BCELoss(fake_probs, noisy_fake) + BCELoss(real_probs, noisy_real)
                D_loss.backward(retain_graph=True)
                D_optimizer.step()

            # Train Generator
            G_optimizer.zero_grad()
            fake_probs = D(imgs_gen, recipe_embs)
            G_loss = BCELoss(fake_probs, all_real)
            G_loss.backward()
            G_optimizer.step()
            
            if iepoch % opts.INTV_PRINT_LOSS == 0 and not ibatch:
                print_loss(G_loss, D_loss, iepoch)
            if iepoch % opts.INTV_SAVE_IMG == 0 and not ibatch:
                # Save a training image
                get_img_gen(data, 0, G, iepoch, opts.IMG_OUT_PATH)
                # Save a validation image
                get_img_gen(data, 1, G, iepoch, opts.IMG_OUT_PATH)
            if iepoch % opts.INTV_SAVE_MODEL == 0 and not ibatch:
                print('Saving model...')
                save_model(G, G_optimizer, D, D_optimizer, iepoch, opts.MODEL_OUT_PATH)

    save_model(G, G_optimizer, D, D_optimizer, 'FINAL', opts.MODEL_OUT_PATH)
    print('\a') # Ring the bell to alert the human
    
if __name__ == '__main__':
    main()
