#訓練データに対して自分の決めた確率で間違ったラベルのデータを追加する
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pickle
import random

def confirm_data(train_labels, train_labels_error, error_idxs):
    count=0
    for idx in range(len(train_labels)):
        if train_labels[idx]!=train_labels_error[idx]:
            count+=1
            #print(train_labels[idx],train_labels_error[idx])
    print(len(train_labels))
    print(len(error_idxs))
    print(count/len(train_labels)*100)

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
            error_idxs.append(i)#ラベルがエラーのものだったらその画像のindexを追加してる

    return labels_error, error_idxs

def make_label_list(true_labels, labels_error, error_idxs, trainset):
    print(labels_error)
    label_zero = []
    label_one = []
    label_two = []
    label_three = []
    label_four = []
    label_five = []
    label_six = []
    label_seven = []
    label_eight = []
    label_nine = []
    label_train = []
    for i, label in enumerate(labels_error):
        label_train.append({'label':label.item(), 'true_label':true_labels[i], 'image':trainset.train_data[i]})
        
    for i, label in enumerate(labels_error):
        label =label.item()
        if label == 0:
            label_zero.append({'label':label, 'true_label':true_labels[i], 'image':trainset.train_data[i]})
        elif label == 1:
            label_one.append({'label':label, 'true_label':true_labels[i], 'image':trainset.train_data[i]})
        elif label == 2:
            label_two.append({'label':label, 'true_label':true_labels[i], 'image':trainset.train_data[i]})
        elif label == 3:
            label_three.append({'label':label, 'true_label':true_labels[i], 'image':trainset.train_data[i]})
        elif label == 4:
            label_four.append({'label':label, 'true_label':true_labels[i], 'image':trainset.train_data[i]})
        elif label == 5:
            label_five.append({'label':label, 'true_label':true_labels[i], 'image':trainset.train_data[i]})
        elif label == 6:
            label_six.append({'label':label, 'true_label':true_labels[i], 'image':trainset.train_data[i]})
        elif label == 7:
            label_seven.append({'label':label, 'true_label':true_labels[i], 'image':trainset.train_data[i]})
        elif label == 8:
            label_eight.append({'label':label, 'true_label':true_labels[i], 'image':trainset.train_data[i]})
        elif label == 9:
            label_nine.append({'label':label, 'true_label':true_labels[i], 'image':trainset.train_data[i]})
    return label_zero, label_one, label_two, label_three, label_four, label_five, label_six, label_seven, label_eight, label_nine, label_train

def get_img():
    batch_size=100
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, ), (0.5, ))])
    trainset = torchvision.datasets.MNIST(root='./data_mnist', train=True, download=True, transform=transform)
    train_labels = trainset.train_labels
    train_labels_error, error_idxs = put_error(train_labels, 0.2)
    label_zero, label_one, label_two, label_three, label_four, label_five, label_six, label_seven, label_eight, label_nine, label_train = make_label_list(train_labels, train_labels_error, error_idxs, trainset)
    print(train_labels_error)
    confirm_data(train_labels,train_labels_error, error_idxs)
    
    #trainloaderにtrainsetを入れるとbatchごとによんでくれる
    trainset.labels = train_labels_error
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.MNIST(root='./data_mnist', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))
    return testloader,train_labels_error,trainset, trainloader, label_zero, label_one, label_two, label_three, label_four, label_five, label_six, label_seven, label_eight, label_nine, label_train

if __name__ == '__main__':
    testloader, train_labels_error, trainset, trainloader, label_zero, label_one, label_two, label_three, label_four, label_five, label_six, label_seven, label_eight, label_nine, label_train = get_img()
    print(label_zero)
    
