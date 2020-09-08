import os
import csv
import librosa
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

dataset_folder = 'refined_data'
info_file = 'info.csv'
output_file = 'voices.csv'

def plot_frequencies(data): # to check if there is any noise
    fourier = np.fft.fft(data)

    n = len(data)
    fourier = fourier[0:int(n/2)]
    # scale by the number of points so that the magnitude does not depend on the length
    fourier = fourier / float(n)
    #calculate the frequency at each point in Hz
    freqArray = np.arange(0, (n/2), 1.0) * (rate*1.0/n);
    x = freqArray[freqArray<300] #human voice range
    y = 10*np.log10(fourier)[0:len(x)]
    plt.figure(1,figsize=(20,9))
    plt.plot(x, y, color='teal', linewidth=0.5)
    plt.xlabel('Frequency (Hz)', fontsize=18)
    plt.ylabel('Amplitude (dB)', fontsize=18)
    plt.show()

def get_frequencies(voice_path):
    data, rate = librosa.load(voice_path)

    #plot_frequencies(data)

    #get dominating frequencies in sliding windows of 200ms
    step = int(rate / 5)
    window_frequencies = []
    for i in range(0,len(data),step):
        ft = np.fft.fft(data[i:i+step])
        freqs = np.fft.fftfreq(len(ft)) #fftq tells you the frequencies associated with the coefficients
        imax = np.argmax(np.abs(ft))
        freq = freqs[imax]
        freq_in_hz = abs(freq *rate)
        window_frequencies.append(freq_in_hz)
    filtered_frequencies = [f for f in window_frequencies if 20<f<300 and not 95<f<105]

    return filtered_frequencies

def get_features(frequencies):
    nobs, minmax, mean, variance, skew, kurtosis = stats.describe(frequencies)
    median    = np.median(frequencies)
    mode      = stats.mode(frequencies).mode[0]
    std       = np.std(frequencies)
    low,peak  = minmax
    q75,q25   = np.percentile(frequencies, [75 ,25])
    iqr       = q75 - q25
    return nobs, mean, skew, kurtosis, median, mode, std, low, peak, q25, q75, iqr


print('start reading info')
ages = {}
genders = {}
with open(info_file, 'r') as f:
    for line in f.readlines():
        if len(line) == 0: continue
        parts = line.split(',')
        if len(parts) < 3: continue
        _id = parts[0].replace(' ', '').replace('\n', '').replace('\t', '')
        gender = parts[1].replace(' ', '').replace('\n', '').replace('\t', '')
        age = parts[2].replace(' ', '').replace('\n', '').replace('\t', '')
        genders[_id] = gender
        ages[_id] = age

print('done reading info')

csv_columns = ['id', 'gender', 'age', 'nobs', 'mean', 'skew', 'kurtosis', 'median', 
               'mode', 'std', 'low', 'peak', 'q25', 'q75', 'iqr']
samples = []
_all = 2859
_cnt = 0
for voice in os.listdir(dataset_folder):
    if 'mp3' in voice:
        _cnt += 1
        print(_cnt,'/',_all, voice)
        frequencies = get_frequencies(os.path.join(dataset_folder, voice))
        if len(frequencies) > 10:
            nobs, mean, skew, kurtosis, median, mode, std, low, peak, q25, q75, iqr = get_features(frequencies)
            _id = voice[:voice.find('_')]
            sample_dict = {'id': _id, 'gender': genders[_id], 'age': ages[_id],
                           'nobs':nobs, 'mean':mean, 'skew':skew, 'kurtosis':kurtosis,
                           'median':median, 'mode':mode, 'std':std, 'low': low,
                           'peak':peak, 'q25':q25, 'q75':q75, 'iqr':iqr}
            samples.append(sample_dict)

print('start writing csv')
with open(output_file, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    for data in samples:
        writer.writerow(data)


