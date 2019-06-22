import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.manifold import TSNE
from sklearn.metrics import explained_variance_score
#from dataloader import load_visualize_data

def load_visualize_data(n_sample):
    (_, _), (X_test, y_test) = mnist.load_data()
    X = np.zeros((n_sample * 10, X_test.shape[1], X_test.shape[2]))
    y = np.zeros(n_sample*10)
    for num in range(10):
        dest_indices = np.arange(num*n_sample, (num+1)*n_sample)
        source_indices = np.where(y_test == num)[0][:n_sample]
        X[dest_indices, :, :] = X_test[source_indices, :, :]
        y[dest_indices] = y_test[source_indices]
    return X, y

def main():
    X, y = load_visualize_data(600)#0～9の各ラベルに対して、n_sample(100)個、n_sample=100の場合は1000個サンプルを取り出して並べた
    X /= 255
    X = X.reshape(X.shape[0], -1)
    print(X)
    decomp = TSNE(n_components=2)
    X_decomp = decomp.fit_transform(X)
    
    cmap = get_cmap("tab10")
    for i in range(10):
        marker = "$" + str(i) + "$"
        indices = np.arange(i*100, (i+1)*100)
        plt.scatter(X_decomp[indices, 0], X_decomp[indices, 1], marker=marker, color=cmap(i))
    plt.show()

if __name__ == '__main__':
    main()
