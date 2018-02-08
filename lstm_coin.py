import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from pprint import pprint
import json
import os

np.random.seed(0)


"""
JSON 파일 추출
"""
files = []
IN_PATH = './result/'
for filename in os.listdir(IN_PATH):
    if filename.endswith(".json"):
        files.append(filename)

"""
데이터 읽기
"""
f = []
start_date = 0
end_date = 0

cnt = 0
for file in files:
    if cnt == 0:
        start_date = int(file.split('.')[0])

    print(cnt, 'read:', file)

    dim = []
    with open(IN_PATH + file) as data_:
        dict = json.load(data_)
        dim.append(dict['volume'])
        dim.append(dict['weightedAverage'])

    f.append(dim)

    end_date = int(file.split('.')[0])
    cnt += 1


    if cnt == 10000: ##
        break ##

"""
데이터 생성
"""
len_sequences = cnt # 전체 시계열 길이
window = 288 # 하나의 시계열 데이터 길이

data = []
target = [] # 정답레이블

for i in range(len_sequences - window):
    data.append(f[i: i+window])
    target.append(f[i+window])

X = np.array(data).reshape(len(data), window, 2)
Y = np.array(target).reshape(len(target), 2)

N_train = int(len(data) * 0.9)
N_vali = len(data) - N_train

X_train, X_vali, Y_train, Y_vali = train_test_split(X, Y, test_size=N_vali)

"""
모델 공통 설정
"""
n_in = len(X[0][0]) # 2
n_hidden = 30
n_out = len(Y[0]) # 2

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

"""
모델 불러오기
"""
h5_s = []
IN_PATH = './model/'
if not os.path.exists(IN_PATH):
    os.makedirs(IN_PATH)

for filename in os.listdir(IN_PATH):
    h5_s.append(filename)

h5_cnt = len(h5_s)

if  h5_cnt != 0:
    model = load_model(IN_PATH + h5_s[-1])
    print('import: ', h5_s[-1])

else:
    """
    모델 설정
    """
    model = Sequential()
    model.add(LSTM(n_hidden,
                   kernel_initializer='random_uniform',
                   input_shape=(window, n_in)))
    model.add(Dense(n_out, kernel_initializer='random_uniform'))
    model.add(Activation('linear'))

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer)

"""
모델 학습
"""
epochs = 100
batch_size = 50

model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_vali, Y_vali),
          callbacks=[early_stopping])

"""
모델 저장
"""
h5_cnt += 1
OUT_PATH = './model/'
model.save(OUT_PATH + str(h5_cnt) + '.h5')
print('create: ', str(h5_cnt) + '.h5')

"""
output to use
"""
original = [f[i][1] for i in range(window)]
predicted = [None for i in range(window)]

Z = X[:1] # 가장 앞
for i in range(len_sequences - window + 1):
    z_ = Z[-1:] # 가장 뒤
    y_ = model.predict(z_)
    sequence_ = np.concatenate(
        (z_.reshape(window, n_in)[1:], y_),
        axis=0).reshape(1, window, n_in)
    Z = np.append(Z, sequence_, axis=0)

    print(i, 'pred:', y_)
    predicted.append((y_.reshape(-1))[1])

"""
visualization
"""
plt.rc('font', family='serif')
plt.figure()
plt.plot([f[i][1] for i in range(len_sequences)], linestyle='dotted', color='#aaaaaa') # 원래 그래프
plt.plot(original, linestyle='dashed', color='black') # 예측에 사용한 초기값 (window)
plt.plot(predicted, color='black') # 예측값
plt.show()
