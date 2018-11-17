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

'''
G = Generator()
D = Discriminator()
criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)

for epoch in range(num_epochs):
    for d_index in range(d_steps):
        # 1. Train D on real+fake
        D.zero_grad()

        #  1A: Train D on real
        d_real_data = Variable(d_sampler(d_input_size))
        d_real_decision = D(preprocess(d_real_data))
        d_real_error = criterion(d_real_decision, Variable(torch.ones(1)))  # ones = true
        d_real_error.backward() # compute/store gradients, but don't change params

        #  1B: Train D on fake
        d_gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
        d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
        d_fake_decision = D(preprocess(d_fake_data.t()))
        d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(1)))  # zeros = fake
        d_fake_error.backward()
        d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

    for g_index in range(g_steps):
        # 2. Train G on D's response (but DO NOT train D on these labels)
        G.zero_grad()

        gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
        g_fake_data = G(gen_input)
        dg_fake_decision = D(preprocess(g_fake_data.t()))
        g_error = criterion(dg_fake_decision, Variable(torch.ones(1)))  # we want to fool, so pretend it's all genuine

        g_error.backward()
        g_optimizer.step()  # Only optimizes G's parameters
'''
