import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os

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
            b = 7800
            # if chn == 1:
            #     a = 500
            #     b = 4740
            # elif chn == 2:
            #     a = 3000
            #     b = 2440
            c = 150
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

def pca_data2(data_dict, sub_names):
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


def pca_analysis(X, Y, ch1, ch2):
    pca = PCA(n_components=2)
    pca.fit(X)
    z = pca.transform(X)
    n_comp = pca.components_

    plt.figure()
    colors = ["navy", "yellow"]
    for color, i, target_name in zip(colors, [0, 1], [ch1, ch2]):
        plt.scatter(z[Y == i, 0], z[Y == i, 1], color=color, alpha=0.8, label=target_name)
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.xlabel('feature1')
    plt.ylabel('feature2')
    plt.show()
    print("Done")


if __name__ == '__main__':
    # folder_location = 'D:/HK/Anita/EMG_time_text_file/'
    folder_location = 'D:\MtechProject_2023\EMG_spectrum_ch3,4/'
    channel_names = ['Thumb:ch3', 'Cheek:ch4']
    file_names = get_file_names(folder_location)
    sub_names = []
    data_dict = {}
    spectrum_data_dict = {}
    for file_name in file_names:
        sub_name = file_name[0:-4].lower()
        sub_name = sub_name.replace(',', '_')
        sub_names.append(sub_name)
        spectrum_data_dict[sub_name] = read_emg_spectrum_data(folder_location + file_name)
        # data_dict[sub_name] = read_emg_data_five_channel(folder_location + file_name)

    X, Y = pca_spectrum_data(spectrum_data_dict, sub_names)
    # print(len(data_dict))
    # X, Y = pca_data2(data_dict, sub_names)
    print(len(X))
    pca_analysis(X, Y, channel_names[0], channel_names[1])
    print("Done")
