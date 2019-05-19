import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import pickle
import os.path
import annoy_mnist
#%matplotlib inline

def get_mnist():
    batch_size = 100
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, ), (0.5, ))])
    trainset = torchvision.datasets.MNIST(root='../mnist/data_mnist', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
    testset = torchvision.datasets.MNIST(root='../mnist/data_mnist', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
   
    return trainloader, trainset

def give_trainimg():
    trainset = get_mnist()
    for (img, label) in trainset:
        dog = before_DoG(img)
#k_size->kernel size, k->constant to scale
def DoG(img, k_size, sigma, k):
    sigma_large = sigma * k
    G_small = cv2.GaussianBlur(img, k_size, sigma)
    G_large = cv2.GaussianBlur(img, k_size, sigma_large)
    D = G_small - G_large
    return D

def preprocessing(img):
    img = cv2.resize(img, (28, 28))
    return img

def extract_feature(img_test, img_train):
    # find the keypoints and descriptors
    # 特徴点に対して特徴記述子(descriptor)を計算
    kp_test, des_test = cv2.img_test.detectAndCompute(img_test, None)
    kp_train, des_train = cv2.img_train.detectAndCompute(img_train, None)
    # 記述子を比較して近いものからマッチング
    matches = bf.match(des_test, des_train)
    # distanceは近い程よい
    dist = [m.distance for m in matches]
    ret = sum(dist) / len(dist)
    return(ret, img_train)


def extract_feature(img_test, img_train):
    img_train = img_train.numpy()
    dist = np.linalg.norm(img_test - img_train)
    return(dist, img_train)

# def before_DoG(torch_img):#torch_img->tensor
#     np_images_one = torch_img.numpy()
#     # [c, h, w] => [h, w, c]
#     np_images = np.transpose(np_images_one, (1, 2, 0))
#     np_images = np_images.astype('uint8')
#     k_size = (3,3)
#     sigma = 1.3
#     k = 1.6
#     dog = DoG(np_images, k_size, sigma, k)
#     return(dog, np_images_one)
   
# def get_pass_img(test_miss_mnist, train_mnist_set):
#     ret_list = []
#     for img_num in range(60000):
#         # train_mnist = preprocessing(train_mnist_set.train_data[img_num])
#         train_mnist = train_mnist_set.train_data[img_num]
#         ret, img_train = extract_feature(test_miss_mnist, train_mnist)
#         ret_list.append({'distance' : ret, 'img' : img_train})
#     ret_list_sort = sorted(ret_list, reverse=True, key=lambda x:x['distance']) # distance is sorted
#     return(ret_list_sort)

def subplot(ret_list_top10, test_img, k, num):
    print('this is uncorrect image')
    #plt.imshow(test_img.reshape(28,28))
    plt.imshow(test_img)
    str_file_1 = '%03.f'%(num)+'.png'
    path_file_1 = os.path.join('./uncorrect_img',str_file_1)
    plt.savefig(path_file_1)
    plt.show()
    for i in range(10):
        train_img = ret_list_top10[i]['img']
        score = ret_list_top10[i]['score']
        # fig = plt.figure()
        print('score : ',score)
        if ret_list_top10[i]['num_err_tf']:
            print('noise label')
            print(ret_list_top10[i]['uncorrect_label'])
        #print(train_img.shape)
        #plt.imshow(train_img.reshape(28,28))
        plt.imshow(train_img)
        str_file_2 = '%03.f'%(k)+'.png'
        path_file_2 = os.path.join('./similar_img',str_file_2)
        plt.savefig(path_file_2)
        k = k+1
        plt.show()
        plt.close()

def get_train_err(err_label, correct_label):
     if err_label!=correct_label:
         num_err = 1#if this image's train label is not correct, num_err is 1
     else:
         num_err = 0
         
def seikei(img):
    img = img.numpy()
    img = np.transpose(img, (1,2,0))
    img = img.astype('uint8')
    img = cv2.resize(img, (200,200))
    return(img)
    
def akaze(test_img, train_img_list, err_labels_list, trainset):
    detector = cv2.AKAZE_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #print(type(test_img))
    test_img = seikei(test_img)
    (test_kp, test_des) = detector.detectAndCompute(test_img, None)
    matches_list = []
    for img_num in range(60000):
        train_mnist = train_img_list[img_num]
        train_mnist = seikei(train_mnist)
        (train_kp, train_des) = detector.detectAndCompute(train_mnist, None)
        num_err = get_train_err(err_labels_list[img_num], trainset.train_labels[img_num])
        matches = bf.match(train_des, test_des)
        #dist = [m.distance for m in matches]
        #ret = sum(dist) / len(dist)
        matches_list.append({'score' : len(matches), 'img' : train_mnist, 'num_err_tf' : num_err, 'correct_label' : trainset.train_labels[img_num], 'uncorrect_label' : err_labels_list[img_num]})
        #print(matches_list)
        
    matches_list = sorted(matches_list,  key = lambda x:-x['score'])
    return(matches_list, test_img)
    
def main():
    k=0
    num=0
    train_img_list = []
    #train_label_list = []
    with open('../mnist/list.txt', 'rb') as list_result:
        data = pickle.load(list_result)
    with open('../mnist/err_trainlabel.txt', 'rb') as err_list:
        err_labels_list = pickle.load(err_list)
    trainloader, trainset = get_mnist()
    for batch_idx, (images, labels) in enumerate(trainloader):
        for idx in range(len(images)):
            train_img_list.append(images[idx])
    for dec in data:
        if dec['label'] != dec['predict']:
            torch_img = torch.tensor(dec['image'])
            # #for DoG
            # test_miss_mnist, not_dog_img = before_DoG(torch_img)
            # ret_list_sort = get_pass_img(test_miss_mnist, train_mnist_set)
            # ret_list_top10 = ret_list_sort[:10]
            # subplot(ret_list_top10, not_dog_img)
            
            # #for AKAZE
            # matches,test_img = akaze(torch_img, train_img_list, err_labels_list, trainset)
            # matches = matches[:10]
            # subplot(matches, test_img, k, num)
            # k=k+10
            # num=num+1

            #for annoy
            predict_indexes=annoy_mnist.make_model(torch_img, train_img_list)
            for j, predict_i in enumerate(predict_indexes):
                img = train_img_list[predict_i].numpy()
                img = np.transpose(img,(1,2,0))
                img = img.astype('uint8')
                img = cv2.resize(img, (200,200))
                plt.imshow(img)
                plt.show()
            
    
if __name__ == '__main__':
    main()
