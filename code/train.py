# Based loosely off https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
import torch
from torch.autograd import Variable
import torch.optim
import torch.nn
import torch.utils.data
from dataset import GANstronomyDataset
import os
from model import Generator, Discriminator

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# OPTIONS
IMAGE_SIZE = 64
BATCH_SIZE = 16
DATA_PATH = os.path.abspath('../temp/data100/data.pkl')
NUM_EPOCHS = 1
CRITERION = torch.nn.BCELoss()
EMBED_SIZE = 1024
ADAM_LR = 0.001
ADAM_B = (0.9, 0.999)

def main():
    # Load the data
    data = GANstronomyDataset(DATA_PATH)
    data_loader = torch.utils.data.DataLoader(data,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)
    num_classes = data.num_classes()

    # Instantiate the models
    G = Generator(EMBED_SIZE, num_classes).to(device)
    G_optimizer = torch.optim.Adam(G.parameters(), lr=ADAM_LR, betas=ADAM_B)

    D = Discriminator(num_classes).to(device)
    D_optimizer = torch.optim.Adam(D.parameters(), lr=ADAM_LR, betas=ADAM_B)

    for iepoch in range(NUM_EPOCHS):
        for ibatch, data_batch in enumerate(data_loader):
            # recipe_embs is [batch_size, EMBED_SIZE]
            # imgs is [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]
            recipe_ids, recipe_embs, img_ids, imgs, classes = data_batch

            # Set up Variables
            batch_size = imgs.shape[0]
            recipe_embs = Variable(recipe_embs.type(FloatTensor)).to(device)
            imgs = Variable(imgs.type(FloatTensor)).to(device)
            classes = Variable(classes.type(LongTensor)).to(device)
            classes_one_hot = Variable(FloatTensor(batch_size, num_classes).zero_().scatter_(1, classes.view(-1, 1), 1)).to(device)

            # Adversarial ground truths
            all_real = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False).to(device)
            all_fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False).to(device)
            
            # Train Generator
            G_optimizer.zero_grad()
            imgs_gen = G(recipe_embs, classes_one_hot)
            fake_probs = D(imgs_gen, classes_one_hot) # TODO: maybe use MSE loss to condition generator
            G_loss = CRITERION(fake_probs, all_fake)
            G_loss.backward()
            G_optimizer.step()

            # Train Discriminator
            D_optimizer.zero_grad()
            fake_probs = D(imgs_gen.detach(), classes_one_hot)
            real_probs = D(imgs, classes_one_hot)
            D_loss = (CRITERION(fake_probs, all_fake) + CRITERION(real_probs, all_real)) / 2
            D_loss.backward()
            D_optimizer.step()
            
            
if __name__ == '__main__':
    main()
