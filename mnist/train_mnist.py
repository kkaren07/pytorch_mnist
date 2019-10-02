import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from model import TheModelClass
import load_data
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
import t_sne

def train(epochs,train_loader):
    global model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=0.0005, momentum=0.99, nesterov=True)
    #pred_label=[]
    
    #train_loader = load_data.trainloader
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 100 == 99:# print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, batch_idx + 1, running_loss / 100))
                running_loss = 0.0
        
        print('Finished Training')
        #test()
    torch.save(model.state_dict(), 'weight')

def test2(model, testloader):
    param = torch.load('weight')
    model.load_state_dict(param)
    model = model.eval()
    result = []
    i=0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            #1行ごとの最大値,1番accuracyが高いlabelを1batch100slicesのimageからget
            _, predicted = torch.max(outputs, 1)
            label_testfail = make_list(images, labels, predicted)
            print(len(images))
            print(i)
            i=i+1
            for idx in range(len(images)):
                result.append({"image" : images[idx], "label": labels[idx].item(), "predict": predicted[idx].item()})
    #to_dog(result)
    return result,label_testfail

#tsneに渡す用のtrainと推論に失敗したtestのリストを作成
def make_list(images, labels, test_predicted):
    label_testfail = []
    for idx in range(len(images)):
        label_testfail.append({'label':test_predicted[idx].item(), 'true_label':labels[idx].item(), 'image':images[idx]})
    return(label_testfail)

def output_error(result):
    for dec in result:
        if dec["label"] != dec["predict"]:
            np_images = dec["image"].numpy()
            plt.imshow(np_images.reshape(28,28))
            print("label", dec["label"], " vs ", "predict: ", dec["predict"])
            plt.show()

def to_dog(result):
    result_list = []
    f = open('list.txt', 'wb')
    for dec in result:
        #img_value = dec["image"].values()
        #print(dec['image'])
        result_list.append({"image":dec["image"].tolist(), "label":dec["label"], "predict":dec["predict"]})
    pickle.dump(result_list, f)
    
def test():#もしtestだけしたい時はtest(model)にして保存したモデルを入れる
    predicted_list = []
    param = torch.load('weight')
    model.load_state_dict(param)
    model = model.eval()
    total = 0
    correct = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in load_data.testloader:
            images, labels = data
            outputs = model(images)
            #1行ごとの最大値,1番accuracyが高いlabelを1batch100slicesのimageからget
            _, predicted = torch.max(outputs, 1)
            predicted_list.append(predicted)
            c = (predicted == labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(100):#range(batch_size)
                label = labels[i]#label:0~9
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            load_data.classes[i], 100 * class_correct[i] / class_total[i]))
    #total accuracy
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    imgshow(predicted_list)

def imgshow(predicted_list):
    #img = images / 2 + 0.5     # unnormalize
    #1batch from test_data
    dataiter = iter(load_data.testloader)
    images, labels = dataiter.next()
    for image in range(len(images)):
        np_images = images[image].numpy()
        plt.imshow(np_images.reshape(28,28))
        plt.show()
        print('label:',predicted_list[0][image])
       
    #print(' '.join('%s' % load_data.classes[predicted[j]] for j in range(100)))
    #print('pred:%d' % (load_data.classes[pled_label[j]] for j in range(100)), 'ans:%d'%(load_data.classes[labels[i]] for i in range(100)))
def sum_list(list_a, list_b):
    list_sum = list_a.append(list_b)
    print(list_sum)
    return(list_sum)
    
if __name__ == '__main__':
    testloader, trainlabel_err, trainset, trainloader, label_zero, label_one, label_two, label_three, label_four, label_five, label_six, label_seven, label_eight, label_nine, label_train= load_data.get_img()
    file_n = open('err_trainlabel.txt','wb')
    pickle.dump(trainlabel_err, file_n)
    epochs = 2
    model = TheModelClass()
    train(epochs,trainloader)
    #test(model)
    result,label_testfail = test2(model, testloader)
    #output_error(result)
    list_for_tsne = sum_list(label_testfail, label_train)
    t_sne.main(list_for_tsne)
