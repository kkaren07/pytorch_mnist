import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import load_data
import matplotlib.pyplot as plt

def main():
    #img = images / 2 + 0.5     # unnormalize
    #1batch from test_data
    images, labels = iter(load_data.testloader).next()
    npimg =  torchvision.utils.make_grid(images, nrow=10, padding=1).numpy()
    # [c, h, w] => [h, w, c]
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
   # print(' '.join('%s' % load_data.classes[predicted[j]] for j in range(100)))
    #print('pred:%d' % (load_data.classes[pled_label[j]] for j in range(100)), 'ans:%d'%(load_data.classes[labels[i]] for i in range(100)))

if __name__ == '__main__':
    main()
