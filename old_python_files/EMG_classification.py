import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def read_emg_data_single_channel(file_location):
    data = []
    with open(file_location) as f:
        for line in f:
            a = line.strip().split()
            data.append((float(a[0]), float(a[1])))

    print(f'File Location: {file_location}, data size: {len(data)}')
    return data


def pca_data(data_dict, file_names):
    X = []
    Y = []
    file_names_tuple = [(file_names[0], file_names[1], file_names[2]), (file_names[3], file_names[4], file_names[5])]

    for j, fn_tp in enumerate(file_names_tuple):
        for i in range(30000, 40000):
            X.append(np.array([data_dict[fn_tp[0]][i][1], data_dict[fn_tp[1]][i][1], data_dict[fn_tp[2]][i][1]]))
            Y.append(j)

    print(f"Length X: {len(X)}, Y: {len(Y)}")
    return np.array(X), np.array(Y)


def pca_data2(data_dict, file_names):
    X = []
    Y = []
    file_names_tuple = [(file_names[0], file_names[1], file_names[2]), (file_names[3], file_names[4], file_names[5])]

    for j, fn in enumerate(file_names):
        a = 50000
        b = 120000
        if j == 2:
            a = 0
            b = 43000
        for k in range(a, b, 500):
            temp = []
            y = []
            for i in range(500):
                temp.append(data_dict[fn][(k + i)][1])
            # y.append(j // 3)
            X.append(np.array(temp))
            Y.append(j//3)
    print(f"Length X: {len(X)}, Y: {len(Y)}")
    return np.array(X), np.array(Y)


if __name__ == '__main__':
    folder_location = 'D:\M.Tech\EMG_picks_results\EMG_Export_data_3subjects_3muscles_nd classification code/'
    file_names = ['thumb_p1', 'thumb_p2', 'thumb_p3', 'leg_p1', 'leg_p2', 'leg_p3']
    data_dict = {}
    for file_name in file_names:
        data_dict[file_name] = read_emg_data_single_channel(folder_location + file_name + '.txt')

    print(len(data_dict))
    X, Y = pca_data2(data_dict, file_names)
    # digits = datasets.load_digits()
    # d = digits.data
    print(len(X))

    pca = PCA(n_components=2)
    pca.fit(X)
    z = pca.transform(X)
    n_comp = pca.components_

    plt.figure()
    # plt.scatter(z[:, 0], z[:, 1], c=Y)
    # plt.colorbar()
    colors = ["navy", "yellow"]
    for color, i, target_name in zip(colors, [0, 1], ['thumb', 'leg']):
        plt.scatter(z[Y == i, 0], z[Y == i, 1], color=color, alpha=0.8, label=target_name)
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.show()
