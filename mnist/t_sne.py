#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.manifold import TSNE
import load_data
import file_save
from sklearn import datasets
from matplotlib.cm import get_cmap
from sklearn.cluster import KMeans
import os
import json


def k_means(img_list,X_reduced,true_list):
    K = 10
    kmeans = KMeans(n_clusters=K).fit(img_list)
    print('log2')
    pred_label = kmeans.predict(img_list)
    # それぞれに与える色を決める。
    color_codes = {0:'#ff0000', 1:'#ff8d03', 2:'#fcfc00', 3:'#35ff03', 4:'#106e2e', 5:'#05e0fc', 6:'#1033e3', 7:'#500fd4', 8:'#e700f7', 9:'#520617'}
    # サンプル毎に色を与える。
    colors = [color_codes[x] for x in pred_label]
    marker = [('$'+str(int(i))+'$') for i in true_list]
    #marker = "$" + str(int(true_label)) + "$"
    print('log3')
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.8, marker=marker, color=colors)
    plt.show()

def main(labels):
    path_w = 'data/tsne.txt'
    path_img = 'data/img.txt'
    img_list=[]
    label_list=[]
    true_list=[]
    #i=0
    for dec in labels:
        # if i ==0:
        #     print(dec['image'].flatten())
        #     print(dec['image'].flatten().shape)
        # i = i+1
        img_list.append(dec['image'].flatten().numpy())
        label_list.append(dec['label'])
        true_list.append(dec['true_label'].item())
    # for batch_idx, (images, labels) in enumerate(trainset):
    #     img_list.append(images.flatten().numpy())
    # print(len(img_list))
    img_list = np.array(img_list)
    X_reduced = TSNE(n_components=3).fit_transform(img_list)#tsneにかけたあと正規化してる
    file_save.main_vec(X_reduced)
    file_save.main_img(img_list)

    # for line in img_list:
    #     if not os.path.isfile(path_img):
    #         with open(path_img, mode='w') as f:
    #             f.write(str(line))
    #             f.write('\n')
    #     else:
    #         with open(path_img, mode='a') as f:
    #             print(line)
    #             print('hoge')
    #             f.write(str(line))
    #             f.write('\n')   

    #k_means(img_list, X_reduced, true_list)
    cmap = get_cmap('tab10')
    for i,true_label in enumerate(true_list):
        marker = "$" + str(int(true_label)) + "$"
        plt.scatter(X_reduced[i, 0], X_reduced[i, 1], marker=marker, color=cmap(int(true_label)))#画像それぞれを次元削減したから[i,0][i,1]にしてる
    #plt.colorbar()
    plt.show()
    
if __name__ == '__main__':
    testloader, train_labels_error, trainset, trainloader, label_zero, label_one, label_two, label_three, label_four, label_five, label_six, label_seven, label_eight, label_nine, sec_list= load_data.get_img()
    main(sec_list)
    #sklearn_mnist()
