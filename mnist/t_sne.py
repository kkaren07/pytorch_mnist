#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import load_data
from sklearn import datasets
from matplotlib.cm import get_cmap
from sklearn.cluster import KMeans

def k_means(img_list,X_reduced,true_list):
    K = 5
    kmeans = KMeans(n_clusters=K).fit(img_list) 
    pred_label = kmeans.predict(img_list)
    # それぞれに与える色を決める。
    color_codes = {0:'#00FF00', 1:'#FF0000', 2:'#0000FF', 3:'#faff00', 4:'#d400ff'}
    # サンプル毎に色を与える。
    colors = [color_codes[x] for x in pred_label]
    marker = [('$'+str(int(i))+'$') for i in true_list]
    #marker = "$" + str(int(true_label)) + "$"
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.8, marker=marker, color=colors)
    plt.show()

def main(labels):
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
    X_reduced = TSNE(n_components=2)
    X_reduced = X_reduced.fit_transform(img_list)
    k_means(img_list, X_reduced, true_list)
    cmap = get_cmap('tab10')
    for i,true_label in enumerate(true_list):
        marker = "$" + str(int(true_label)) + "$"
        plt.scatter(X_reduced[i, 0], X_reduced[i, 1], marker=marker, color=cmap(int(true_label)))#画像それぞれを次元削減したから[i,0][i,1]にしてる
    #plt.colorbar()
    plt.show()
    
if __name__ == '__main__':
    testloader, train_labels_error, trainset, trainloader, label_zero, label_one, label_two, label_three, label_four, label_five, label_six, label_seven, label_eight, label_nine = load_data.get_img()

    main(label_one)
    #sklearn_mnist()
