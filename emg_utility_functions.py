import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
from scipy import integrate


def read_emg_spectrum_data(file_location, channels=2):
    data = []
    with open(file_location) as f:
        for line in f:
            a = line.strip().split()
            temp_list = []
            for chn in range(channels + 1):
                temp_list.append(float(a[chn]))
            data.append(tuple(temp_list))
    print(f'File Location: {file_location}, data size: {len(data)}')
    return data


def pca_spectrum_data(spectrum_data_dict, sub_names):
    X, Y = [], []
    for chn in range(2):
        for j, fn in enumerate(sub_names):
            a = 200
            b = 8000
            # if chn == 1:
            #     a = 500
            #     b = 4740
            # elif chn == 2:
            #     a = 3000
            #     b = 2440
            c = 20
            for k in range(a, b, c):
                temp = []
                for x in range(c):
                    temp.append(spectrum_data_dict[fn][(k + x)][chn + 1])
                X.append(np.array(temp))
                Y.append(chn)
    print(f"Length X: {len(X)}, Y: {len(Y)}")
    return np.array(X), np.array(Y)


def read_emg_data_five_channel(file_location):
    data = []
    with open(file_location) as f:
        for line in f:
            a = line.strip().split()
            data.append((float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4]), float(a[5])))
    print(f'File Location: {file_location}, data size: {len(data)}')
    return data


def pca_data_time(data_dict, sub_names):
    X = []
    Y = []
    # sub_sample_size = 60000
    for sub in range(2):
        for j, fn in enumerate(sub_names):
            a = 0
            b = 300000
            # a = 0 + sub * sub_sample_size
            # b = a + sub_sample_size
            print(a, b)
            c = 200
            for k in range(a, b, c):
                temp = []
                for x in range(c):
                    temp.append(data_dict[fn][(k + x)][sub + 1])
                X.append(np.array(temp))
                Y.append(sub)

    print(f"Length X: {len(X)}, Y: {len(Y)}")
    return np.array(X), np.array(Y)


def get_file_names(folder_location):
    names = []
    for file in os.listdir(folder_location):
        if file.endswith(".txt"):
            names.append(file)
    return names


def pca_analysis(X, Y, ch, n_features=3):
    pca = PCA(n_components=n_features)
    pca.fit(X)
    z = pca.transform(X)
    n_comp = pca.components_
    ch_label = []
    for j, c in enumerate(ch):
        ch_label.append(j)

    fig = plt.figure()
    colors = ["navy", "cyan", "magenta", "green", "red", "yellow", "blue", "black"]
    colors_ch = colors[0:len(ch)]
    if n_features == 3:
        ax = fig.add_subplot(111, projection='3d')
        for color, i, target_name in zip(colors_ch, ch_label, ch):
            ax.scatter(z[Y == i, 0], z[Y == i, 1], z[Y == i, 2], color=color, alpha=0.8,
                       label=target_name)
        ax.set_xlabel('principal component 1')
        ax.set_ylabel('principal component 2')
        ax.set_zlabel('principal component 3')
    elif n_features == 2:
        for color, i, target_name in zip(colors_ch, ch_label, ch):
            plt.scatter(z[Y == i, 0], z[Y == i, 1], color=color, alpha=0.8,
                        label=target_name)
        plt.xlabel('principal component 1')
        plt.ylabel('principal component 2')
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title('Principal Component Analysis(PCA) for EMG Data: Dimensionality reduction')
    # plt.show()
    print("Done")
    return z


def extract_feature(d: dict, subject_names, channels=2, start_point=200, stop_point=7000, window_size=200):
    X, Y = [], []
    feature_vectors, feature_label = [], []
    for chn in range(channels):
        for j, fn in enumerate(subject_names):
            feature_vector = []
            # start_point, stop_point, window_size = 200, 5000, 300
            for k in range(start_point, stop_point, window_size):
                temp = []
                freq = []
                for i in range(window_size):
                    temp.append(d[fn][(k + i)][chn + 1])
                    freq.append(d[fn][(k + i)][0])
                feature_vector.append(integrate.simps(np.array(temp), np.array(freq)))
                X.append(np.array(temp))
                Y.append(chn)
            feature_vectors.append(np.array(feature_vector))
            feature_label.append(chn)
    feature_vectors_norm = []
    for fv in feature_vectors:
        feature_vectors_norm.append(fv / sum(fv))
    print(f"Feature matrix:\n{feature_vectors_norm}")
    print("No of samples: {}".format(len(feature_vectors_norm)))
    print(f"Length Feature Vector: {len(feature_vectors_norm[0])}")
    print(f"Length Feature Label: {len(feature_vectors_norm[0])}")
    return np.array(X), np.array(Y), np.array(feature_vectors_norm), np.array(feature_label)


def get_data_dict(folder_location, channels=2):
    file_names = get_file_names(folder_location)
    sub_names = []
    spectrum_data_dict = {}
    for file_name in file_names:
        sub_name = file_name[0:-4].lower()
        sub_name = sub_name.replace(',', '_')
        sub_names.append(sub_name)
        spectrum_data_dict[sub_name] = read_emg_spectrum_data(folder_location + file_name, channels)
    return sub_names, spectrum_data_dict
