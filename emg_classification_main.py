from emg_utility_functions import get_data_dict, extract_feature, pca_analysis
from plot_emg_svm import emg_svm
import numpy as np

if __name__ == '__main__':
    folder_location_ch1_2 = 'D:/MtechProject_2023/EMG_spectrum_time_data_text_files/EMG_spectrum_ch1_2/'
    channel_names_ch1_2 = ['ch1: Leg', 'ch2: Bicep']
    folder_location_ch3_4 = 'D:/MtechProject_2023/EMG_spectrum_time_data_text_files/EMG_spectrum_ch3,4/'
    channel_names_ch3_4 = ['ch3: Thumb', 'ch4: Cheek']

    folder_location_ch5 = 'D:/MtechProject_2023/EMG_spectrum_time_data_text_files/EMG_spectrum_ch5/'
    channel_names_ch5 = ['ch5: Eye']
    # start_point: this is stating frequency , freq domain sampling rate : 1/0.06, start freq: 500 * 0.06 = 30Hz
    # feature width = window_size*0.06 , ex: 300 * 0.06 = 18Hz
    # end point: 5000*0.06 = 300Hz
    start_point, stop_point, window_size = 500, 5000, 300
    pca_features = 2

    sub_names_ch1_2, spectrum_data_dict_ch1_2 = get_data_dict(folder_location_ch1_2)
    X_ch1_2, Y_ch1_2, feature_vectors_norm_ch_1_2, feature_label_ch1_2 = extract_feature(spectrum_data_dict_ch1_2,
                                                                                         sub_names_ch1_2, 2,
                                                                                         start_point,
                                                                                         stop_point, window_size)
    sub_names_3_4, spectrum_data_dict_ch3_4 = get_data_dict(folder_location_ch3_4)
    X_ch3_4, Y_ch3_4, feature_vectors_norm_ch3_4, feature_label_ch3_4 = extract_feature(spectrum_data_dict_ch3_4,
                                                                                        sub_names_3_4, 2, start_point,
                                                                                        stop_point, window_size)

    sub_names_5, spectrum_data_dict_ch5 = get_data_dict(folder_location_ch5)
    X_ch5, Y_ch5, feature_vectors_norm_ch5, feature_label_ch5 = extract_feature(spectrum_data_dict_ch5,
                                                                                sub_names_5, 1,
                                                                                start_point, stop_point, window_size)

    feature_vectors_norm_ch_1_to_4 = np.append(feature_vectors_norm_ch_1_2, feature_vectors_norm_ch3_4, axis=0)
    feature_label_ch_1_to_4 = np.append(feature_label_ch1_2, feature_label_ch3_4 + 2, axis=0)
    channel_names_ch_1_to_4 = channel_names_ch1_2 + channel_names_ch3_4

    feature_vectors_norm_ch_1_to_5 = np.append(feature_vectors_norm_ch_1_to_4, feature_vectors_norm_ch5, axis=0)
    feature_label_ch_1_to_5 = np.append(feature_label_ch_1_to_4, feature_label_ch5 + 4, axis=0)
    channel_names_ch_1_to_5 = channel_names_ch_1_to_4 + channel_names_ch5

    feature_vectors_norm_ch_3_to_5 = np.append(feature_vectors_norm_ch3_4, feature_vectors_norm_ch5, axis=0)
    feature_label_ch_3_to_5 = np.append(feature_label_ch3_4, feature_label_ch5 + 2, axis=0)
    channel_names_ch_3_to_5 = channel_names_ch3_4 + channel_names_ch5

    # emg_data, emg_label, emg_channels = feature_vectors_norm_ch_1_2, feature_label_ch1_2, channel_names_ch1_2
    # emg_data, emg_label, emg_channels = feature_vectors_norm_ch3_4, feature_label_ch3_4, channel_names_ch3_4
    # emg_data, emg_label, emg_channels = feature_vectors_norm_ch_3_to_5, feature_label_ch_3_to_5, channel_names_ch_3_to_5
    emg_data, emg_label, emg_channels = feature_vectors_norm_ch_1_to_4, feature_label_ch_1_to_4, channel_names_ch_1_to_4

    emg_data_pca_components = pca_analysis(emg_data, emg_label, emg_channels, pca_features)
    if pca_features == 2:
        emg_svm(emg_data_pca_components, emg_label, 0.8)
    print("Done")

