from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
# import scipy
from scipy.signal import butter, filtfilt
import math


def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba', fs=None)
    y = filtfilt(b, a, data)
    return y

def draw_power_spectral(data, time_step):
    ps = np.abs(np.fft.fft(data)) ** 2
    freqs = np.fft.fftfreq(no_of_points, time_step)
    idx = np.argsort(freqs)
    return freqs[idx], ps[idx]

time_data = []
ch1_data = []
# ch2_data = []
forearm_data = []
avg_noise = 0
# "C:\Users\anita\OneDrive\Desktop\forearm_front_ch1_all.txt"
with open("C:/Users/anita/OneDrive/Desktop/EMG_signal_2000_sampling/Thumb_muscle_up_ch1_2000_all.txt") as f:
    for line in f:
        # print(line)
        a = line.strip().split()
        time_data.append(float(a[0]))
        ch1_data.append(float(a[1]))
        # ch2_data.append(float(a[2]))
        # avg_noise = avg_noise + abs(float(a[1]))

no_of_points = 1000
# start = 0
# print(time_data[0:10], ch1_data[0:10])

Ts = time_data[1] - time_data[0]
Ts = round(Ts * 10000, 2) / 10000
time_per_grid = 0.25  # in seconds
no_of_grids = 10
window_duration = no_of_grids * time_per_grid
no_of_sample_points_in_window_duration = int((window_duration // Ts) + 1)
start_time_offset = time_data[0]
mid_time = 13 - start_time_offset
start_time = mid_time - 5 * time_per_grid
if mid_time <= 0:
    start = 0
else:
    start = int((start_time // Ts))
no_of_points = no_of_sample_points_in_window_duration
fs = math.ceil(1 / Ts)
print(f'Ts = {Ts}')
print(f'fs = {fs}')
time_data_current_window = time_data[start: start + no_of_points]
un_filtered_data_in_current_window = ch1_data[start: start + no_of_points]
filtered_data_in_current_window = butter_lowpass_filter(data=un_filtered_data_in_current_window, cutoff=225, fs=fs,
                                                        order=2)

plt.plot(time_data_current_window, un_filtered_data_in_current_window)
plt.plot(time_data_current_window, filtered_data_in_current_window)

plt.xlabel("Time in seconds")
plt.ylabel("EMG sample value in mV")
plt.title("EMG signal in time domain")
plt.grid()
plt.legend(['unfiltered data', 'filtered data with low pass'])
plt.figure()
freq_idx, ps_idx = draw_power_spectral(un_filtered_data_in_current_window, Ts)
plt.plot(freq_idx, ps_idx)

freq_idx, ps_idx = draw_power_spectral(filtered_data_in_current_window, Ts)
plt.plot(freq_idx, ps_idx)

plt.xlabel("Frequency in Hz")
plt.ylabel("power sample value in micro V^2")
plt.title("Spectrum of EMG data")
plt.grid()
plt.legend(['unfiltered data', 'filtered data with low pass'])
# plt.legend()
plt.show()
