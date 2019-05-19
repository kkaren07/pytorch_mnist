from annoy import AnnoyIndex
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2

def seikei(img):
    img = img.numpy()
    img = np.transpose(img, (1,2,0))
    
    return(img)

def make_model(test_img, train_list):
    dim = 784 #mnist-1>28*28
    n_tree = 30#parameter
    model = AnnoyIndex(dim)
    for i in range(len(train_list)):
        train_img = train_list[i]
        train_img = seikei(train_img)
        train_img = np.ndarray.flatten(train_img)
        model.add_item(i, train_img)
    test_img = seikei(test_img)
    test_img = np.ndarray.flatten(test_img)
    model.build(n_tree)
    model.save("mnist-30tree.ann")
    predict_indexes=get_neighboor(model, test_img, train_list)
    return(predict_indexes)

def get_neighboor(model, test_img, train_list):
    predict_indexes = model.get_nns_by_vector(test_img, 10, search_k=150)#近傍から10こ取り出す
    return(predict_indexes)
    # for j, predict_i in enumerate(predict_indexes):
    #     img = seikei(train_list[predict_i])
    #     img = img.astype('uint8')
    #     img = cv2.resize(img, (200,200))
    #     plt.imshow(img)
    #     plt.show()


