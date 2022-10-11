import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture as GMM 
from mfcc_coeff import extract_features
import warnings
warnings.filterwarnings("ignore")

source   = "samples\\"   

dest = "model\\"    
train_list = "training_sample_list.txt"        
file_paths = open(train_list,'r')

count = 1
# 特徵提取
features = np.asarray(())
for path in file_paths:    
    path = path.strip()   
    
    # 讀檔
    sr,audio = read(source + path)
    
    # mfcc_coeff.py 提取特徵
    vector   = extract_features(audio,sr)
    
    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
        
    # 模型訓練
    if count == 3:    
        gmm = GMM(n_components = 16, n_iter = 200, covariance_type='diag',n_init = 3)
        gmm.fit(features)
        
        # 儲存 個人 gmm model
        picklefile = path.split("-")[0]+".gmm"
        cPickle.dump(gmm,open(dest + picklefile,'wb'))
        print ('+ modeling completed for speaker:',picklefile," with data point = ",features.shape)
        features = np.asarray(())
        count = 0
    count = count + 1