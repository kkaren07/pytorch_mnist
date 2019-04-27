import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pickle
import random

##MNIST
# Hyperparameters

# labels: tensor put_error(labels, p) return tensor with error

# 確率pで、label以外の要素を等しい確率で返す　1 - pでlabelを返す
def change_label(label, p):
    if random.random() <= p:# pの確率でtrueになる
       un_labels = list(range(10))
       un_labels.remove(label)
       #[0,...9]/label のどれかを等しい確率で返す
       return random.choice(un_labels)
    else:
        return label
    
def put_error(labels, p):
    labels_error = labels.clone()
    error_idxs = []
    for i, label in enumerate(labels):
        label_with_error =  change_label(label, p)
        labels_error[i] = label_with_error
        if label_with_error != label:
            error_idxs.append(i)

    return labels_error, error_idxs

batch_size=100
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ))])
trainset = torchvision.datasets.MNIST(root='./data_mnist', train=True, download=True, transform=transform)
train_labels =trainset.train_labels
train_labels_error, error_idxs = put_error(train_labels, 0.2)
print(train_labels_error)
# print(list(b)) 
#print(b[4][0])
#trainloaderにtrainsetを入れるとbatchごとによんでくれる


trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data_mnist', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))
