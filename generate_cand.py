import os
import joblib
import numpy as np
import tqdm

imdy_train = []
for item in os.listdir('data/imdy_train'):
    if 'cand' not in item: 
        nf = joblib.load(f'data/imdy_train/{item}')['mvel'].shape[0]
        imdy_train.append((f'imdy_train/{item}', nf))
joblib.dump(imdy_train, f'data/imdy_train/cand.pkl')
imdy_test = []
for item in os.listdir('dataimdy_test'):
    if 'cand' not in item: 
        nf = joblib.load(f'data/imdy_test/{item}')['mvel'].shape[0]
        imdy_test.append((f'imdy_test/{item}', nf))
joblib.dump(imdy_train, f'data/imdy_test/cand.pkl')