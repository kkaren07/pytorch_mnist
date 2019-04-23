import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
##VOC2012
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
# trainset = torchvision.datasets.VOCDetection(root='./data', year='2012', image_set='train', download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=24,
#                                           shuffle=True, num_workers=2)
# testset = torchvision.datasets.VOCDetection(root='./data', year='2012', image_set='val', download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=32,
#                                          shuffle=False, num_workers=2)
# classes = ('background','aeroplane','bicycle','bird','boad','bottle','bus','car','cat','chair','cow','dining table','dog','horse','motor bike','person','potted plant','sheep','sofa','train','tv/monitor','void')

##MNIST
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ))])
trainset = torchvision.datasets.MNIST(root='./data_mnist', train=True,download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=100,shuffle=True,num_workers=2)

testset = torchvision.datasets.MNIST(root='./data_mnist', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))
