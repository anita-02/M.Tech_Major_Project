import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os


def read_emg_data_five_channel(file_location):
    data = []
    with open(file_location) as f:
        for line in f:
            a = line.strip().split()
            data.append((float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4]), float(a[5])))
    print(f'File Location: {file_location}, data size: {len(data)}')
    return data


def pca_data2(data_dict, sub_names):
    X = []
    Y = []
    # sub_sample_size = 60000
    for channel in range(2):
        for j, fn in enumerate(sub_names):
            a = 0
            b = 300000
            # a = 0 + channel * sub_sample_size
            # b = a + sub_sample_size
            print(a, b)
            c = 200
            for k in range(a, b, c):
                temp = []
                for x in range(c):
                    temp.append(data_dict[fn][(k + x)][channel + 1])
                X.append(np.array(temp))
                Y.append(channel)

    print(f"Length X: {len(X)}, Y: {len(Y)}")
    return np.array(X), np.array(Y)


def get_file_names(folder_location):
    names = []
    for file in os.listdir(folder_location):
        if file.endswith(".txt"):
            names.append(file)
    return names


if __name__ == '__main__':
    folder_location = 'D:\MtechProject_2023\EMG_Time_5_subs/'
    file_names = get_file_names(folder_location)
    sub_names = []
    data_dict = {}
    for file_name in file_names:
        sub_name = file_name[0:-4].lower()
        sub_names.append(sub_name)
        data_dict[sub_name] = read_emg_data_five_channel(folder_location + file_name)

    print(len(data_dict))
    X, Y = pca_data2(data_dict, sub_names)
    print(len(X))

    pca = PCA(n_components=2)
    pca.fit(X)
    z = pca.transform(X)
    n_comp = pca.components_

    plt.figure()
    colors = ["navy", "yellow"]
    for color, i, target_name in zip(colors, [0, 1], ['leg', 'bicep']):
        plt.scatter(z[Y == i, 0], z[Y == i, 1], color=color, alpha=0.8, label=target_name)
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.show()
