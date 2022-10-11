import pyaudio
import librosa
import wave
import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from .mfcc_coeff import extract_features
import time
import csv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
import keras
import joblib
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import tensorflow as tf
from sklearn import svm
from keras.models import load_model

import warnings
warnings.filterwarnings('ignore')

def clasRecog():
    header = 'filename rms spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header = header.split()

    file = open('dtest.csv', 'w', newline='')
    
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    filename='test.wav'
    songname = f'Speaker_Recognition\\samples\\{filename}'
    y, sr = librosa.load(songname, mono=True, duration=30)
    rmse = librosa.feature.rms(y=y)
    # chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    to_append = f'{filename} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
    for e in mfcc:
        to_append += f' {np.mean(e)}'

    file = open('dtest.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())

    df = pd.read_csv('dtest.csv')
    df = df.drop(['filename'],axis=1)

    test_wav = np.array(df.iloc[:, :])
    test_wav = test_wav[0]

    data = pd.read_csv('data9.csv')
    data = data.drop(['filename'],axis=1)

    ddd = np.array(data.iloc[:, :-1])
    ddd = list(ddd)
    ddd.append(test_wav)
    ddd = np.array(ddd)

    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(ddd, dtype = float))

    test_wav_data = X[-1]
    test_wav_data

    model = load_model('Speaker_Recognition\\model\\my_model3.h5')

    tt = []
    tt.append(list(test_wav_data))
    tt = np.array(tt)

    test_wav_data=tt
    layer_name = 'dense3'
    intermediate_layer_model = Model(inputs = model.input,
                                    outputs = model.get_layer(layer_name).output)

    res = intermediate_layer_model.predict(test_wav_data)

    loaded_model = joblib.load(open('Speaker_Recognition\\model\\clf10.pkl', 'rb'))
    res_y = loaded_model.predict(res)
    ans = str(res_y[0])

    if ans == '0':
        ans = "合成聲音"
    elif ans == '1':
        ans = '變聲聲音'
    elif ans == '2':
        ans = '克隆聲音'
    elif ans == '3':
        ans = '真實聲音'
    else:
        ans = '無法識別聲音'
    
    print(ans)
    return ans
    time.sleep(1.0)