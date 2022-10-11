import pyaudio
import wave
import os
import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from .mfcc_coeff import extract_features
import warnings
import time
warnings.filterwarnings("ignore")

def speakerRecog():
    #Recording Phase 
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "Speaker_Recognition\\samples\\test.wav"
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("錄音中....")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("錄音完畢....")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    source   = "Speaker_Recognition\\samples\\"   
    modelpath = "Speaker_Recognition\\models2\\"
    test_file = "Speaker_Recognition\\testing_sample_list.txt"        
    file_paths = open(test_file,'r')

    gmm_files = [os.path.join(modelpath,fname) for fname in os.listdir(modelpath) if fname.endswith(".gmm")]
    #print (gmm_files)

    #Load the Models
    models = [cPickle.load(open(fname,'rb')) for fname in gmm_files]
    speakers = [fname.split("\\")[-1].split(".gmm")[0] for fname in gmm_files]

    for path in file_paths:   
    
        path = path.strip()   
        print (path)
        sr,audio = read(source + path)
        vector   = extract_features(audio,sr)
        
        log_likelihood = np.zeros(len(models)) 
        
    for i in range(len(models)):
        gmm    = models[i]  # 逐一比對
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()
    
    winner = np.argmax(log_likelihood)
    print ("\tdetected as - ", speakers[winner])
    return speakers[winner]
    time.sleep(1.0)