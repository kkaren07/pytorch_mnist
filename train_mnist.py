import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from net import Net
import load_data
import torch.optim as optim
import matplotlib.pyplot as plt

def parser():
    parser = argparse.ArgumentParser(description='PyTorch MNIST')
    parser.add_argument('--epochs', '-e', type=int, default=2,
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--lr', '-l', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    args = parser.parse_args()
    return args

def main():
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr=0.0005, momentum=0.99, nesterov=True)
    #pred_label=[]
    
    args = parser()
    
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(load_data.trainloader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
                                      
        print('Finished Training')

    torch.save(net.state_dict(), 'weight')

    
    #the_model = TheModelClass(*args, **kwargs)
    #the_model.load_state_dict(torch.load(weight))
    # test
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in load_data.testloader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            #if predicted==labels
            correct += (predicted == labels).sum().item()
            #imgshow(predicted)
            #pred_label.append(predicted)
            
            
    #total accuracy
    print('Accuracy: %2d %%' % (100 * correct/total))

    ## accuracy of each class
    # class_correct = list(0. for i in range(10))
    # class_total = list(0. for i in range(10))
    # with torch.no_grad():
    #     for data in load_data.testloader:
    #         images, labels = data
    #         outputs = net(images)
    #         _, predicted = torch.max(outputs, 1)
    #         c = (predicted == labels).squeeze()
    #         for i in range(4):
    #             label = labels[i]
    #             class_correct[label] += c[i].item()
    #             class_total[label] += 1
    # for i in range(10):
    #     print('Accuracy of %5s : %2d %%' % (
    #         classes[i], 100 * class_correct[i] / class_total[i]))
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    imgshow(predicted)

def imgshow(pred_label):
    #img = images / 2 + 0.5     # unnormalize
    #1batch from test_data
    images, labels = iter(load_data.testloader).next()
    npimg =  torchvision.utils.make_grid(images, nrow=10, padding=1).numpy()
    # [c, h, w] => [h, w, c]
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    print(' '.join('%s' % load_data.classes[predicted[j]] for j in range(100)))
    #print('pred:%d' % (load_data.classes[pled_label[j]] for j in range(100)), 'ans:%d'%(load_data.classes[labels[i]] for i in range(100)))
    
if __name__ == '__main__':
    main()
    
