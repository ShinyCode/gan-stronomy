# Modified from im2recipe-Pytorch/test.py
# Original repo link here: https://github.com/torralba-lab/im2recipe-Pytorch
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.backends.cudnn as cudnn
from data_loader import ImagerLoader # our data_loader
import numpy as np
from trijoint import im2recipe
import pickle
from args import get_parser
import os

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================

torch.manual_seed(opts.seed)
if not opts.no_cuda:
        torch.cuda.manual_seed(opts.seed)

np.random.seed(opts.seed)

def main():
   
    model = im2recipe()
    model.visionMLP = torch.nn.DataParallel(model.visionMLP, device_ids=[0])
    if not opts.no_cuda:
        model.cuda()
   
    print("=> loading checkpoint '{}'".format(opts.model_path))
    checkpoint = torch.load(opts.model_path)
    opts.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(opts.model_path, checkpoint['epoch']))

    # data preparation, loaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # preparing test loader 
    test_loader = torch.utils.data.DataLoader(
        ImagerLoader(opts.img_path,
 	    transforms.Compose([
            transforms.Scale(256), # rescale the image keeping the original aspect ratio
            transforms.CenterCrop(224), # we get only the center of that rescaled
            transforms.ToTensor(),
            normalize,
        ]),data_path=opts.data_path,sem_reg=opts.semantic_reg,partition='temp'),
        batch_size=opts.batch_size, shuffle=False,
        num_workers=opts.workers, pin_memory=(not opts.no_cuda))
    print('Test loader prepared.')

    # run test
    test(test_loader, model)

def test(test_loader, model):
    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(test_loader):
        input_var = [] 
        for j in range(len(input)):
            with torch.no_grad():
                v = torch.autograd.Variable(input[j])
            input_var.append(v.cuda() if not opts.no_cuda else v)
        # compute output
        output = model(input_var[0], input_var[1], input_var[2], input_var[3], input_var[4])
 
        if i == 0:
            data1 = output[1].data.cpu().numpy()
            data3 = target[-1]
        else:
            data1 = np.concatenate((data1,output[1].data.cpu().numpy()),axis=0)
            data3 = np.concatenate((data3,target[-1]),axis=0)

    with open(os.path.join(opts.path_results, 'rec_embeds.pkl'), 'wb') as f:
        pickle.dump(data1, f)
    with open(os.path.join(opts.path_results, 'rec_ids.pkl'), 'wb') as f:
        pickle.dump(data3, f)

if __name__ == '__main__':
    main()
