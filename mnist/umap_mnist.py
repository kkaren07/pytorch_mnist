import pandas as pd
import load_data
import umap
import matplotlib.pyplot as plt
import numpy as np

def main(labels):
    #digits = pd.read_csv("../input/train.csv")
    img_list=[]
    label_list=[]
    true_list=[]
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
    embedding = umap.UMAP(n_neighbors=5,
                          min_dist=0.3,
                          metric='correlation').fit_transform(img_list)
    plt.figure(figsize=(12,12))
    plt.scatter(embedding[:20000, 0], embedding[:20000, 1], 
                edgecolor='none', 
                alpha=0.80, 
                s=10)
    plt.axis('off');
    
if __name__ == '__main__':
    testloader, train_labels_error, trainset, trainloader, label_zero, label_one, label_two, label_three, label_four, label_five, label_six, label_seven, label_eight, label_nine = load_data.get_img()

    main(label_one)
