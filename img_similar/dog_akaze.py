import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import pickle
import os.path

#%matplotlib inline

def get_mnist():
    batch_size = 100
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, ), (0.5, ))])
    trainset = torchvision.datasets.MNIST(root='../mnist/data_mnist', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader

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


def before_DoG(torch_img):#torch_img->tensor
    np_images_one = torch_img.numpy()
    # [c, h, w] => [h, w, c]
    np_images = np.transpose(np_images_one, (1, 2, 0))
    np_images = np_images.astype('uint8')
    k_size = (3,3)
    sigma = 1.3
    k = 1.6
    dog = DoG(np_images, k_size, sigma, k)
    return(dog, np_images_one)
   
def get_pass_img(test_miss_mnist, train_mnist_set):
    ret_list = []
    for img_num in range(60000):
        # train_mnist = preprocessing(train_mnist_set.train_data[img_num])
        train_mnist = train_mnist_set.train_data[img_num]
        ret, img_train = extract_feature(test_miss_mnist, train_mnist)
        ret_list.append({'distance' : ret, 'img' : img_train})
    ret_list_sort = sorted(ret_list, reverse=True, key=lambda x:x['distance']) # distance is sorted
    return(ret_list_sort)

def subplot(ret_list_top10, test_img, k):
    print('this is uncorrect image')
    #plt.imshow(test_img.reshape(28,28))
    plt.imshow(test_img)
    plt.show()
    for i in range(10):
        train_img = ret_list_top10[i]['img']
        distance = ret_list_top10[i]['distance']
        # fig = plt.figure()
        print('distance : ',distance)
        print(train_img.shape)
        #plt.imshow(train_img.reshape(28,28))
        plt.imshow(train_img)
        str_file = "%03.f"%(k)+".png"
        path_file = os.path.join('./similar_img',str_file)
        plt.savefig(path_file)
        k = k+1
        plt.show()
        plt.close()
def akaze(test_img, train_mnist_set):
    detector = cv2.AKAZE_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    #print(type(test_img))
    test_img = test_img.numpy()
    #print(type(test_img))
    test_img = np.transpose(test_img, (1, 2, 0))
    test_img = test_img.astype('uint8')
    test_img = cv2.resize(test_img, (200,200))#detectAndComputeの戻り値がNoneではなくなった！
    (test_kp, test_des) = detector.detectAndCompute(test_img, None)
    matches_list = []
    for img_num in range(60000):
        train_mnist = train_mnist_set[img_num]
        train_mnist = train_mnist.numpy()
        train_mnist = np.transpose(train_mnist, (1, 2, 0))
        train_mnist = train_mnist.astype('uint8')
        train_mnist = cv2.resize(train_mnist, (200,200))
        (train_kp, train_des) = detector.detectAndCompute(train_mnist, None)
        matches = bf.match(train_des, test_des)
        dist = [m.distance for m in matches]
        ret = sum(dist) / len(dist)
        matches_list.append({'distance' : ret, 'img' : train_mnist})
        #print(matches_list)
        
    matches_list = sorted(matches_list, reverse=True, key = lambda x:x['distance'])
    return(matches_list, test_img)
    
def main():
    k=0
    train_img_list = []
    with open('../mnist/list.txt', mode='rb') as list_result:
        data = pickle.load(list_result)
    for dec in data:
        if dec['image'] != dec['predict']:
            torch_img = torch.tensor(dec['image'])
            trainloader = get_mnist()
            for batch_idx, (images, labels) in enumerate(trainloader):
                for idx in range(len(images)):
                    train_img_list.append(images[idx])
            #test_miss_mnist = preprocessing(test_miss_mnist)
            #cv2.imwrite('output.jpg', test_miss_mnist)
            
            # for DoG
            #test_miss_mnist, not_dog_img = before_DoG(torch_img)
            #ret_list_sort = get_pass_img(test_miss_mnist, train_mnist_set)
            #ret_list_top10 = ret_list_sort[:10]
            #subplot(ret_list_top10, not_dog_img)
            
            # for AKAZE
            matches = akaze(torch_img, train_img_list)
            matches,test_img = matches[:10]
            subplot(matches, test_img, k)
            k=k+10
  
    
if __name__ == '__main__':
    main()
