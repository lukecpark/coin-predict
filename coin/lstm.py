import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import adam
import matplotlib.pyplot as plt
from math import sqrt

SEED = 7 # random seed
np.random.seed(SEED)

# main
if __name__ == "__main__":
    """
    데이터 읽기
    """
    PATH = './result/'
    # open, high, low, close, volume
    raw = np.loadtxt(PATH + 'output.csv', delimiter=',', skiprows=1, usecols=[1, 2, 3, 4, 5])

    """
    데이터 정규화
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    nptf = scaler.fit_transform(raw)

    """
    데이터 생성
    """
    seq_len = len(raw) # 전체 시계열 길이
    window = 7 # 하나의 시계열 길이
    dim = 5 # 데이터 차원

    X = []
    Y = []
    for i in range(seq_len - window):
        x_ = nptf[i: i + window]
        y_ = nptf[i + window]
        X.append(x_)
        Y.append(y_)
        # print(x_, '->', y_)

    X = np.array(X)
    Y = np.array(Y)

    # split to train and testing
    N_test = int(len(Y) * 0.2)
    N_val = int((len(Y) - N_test) * 0.1)
    # X_train, X_val, X_test = np.array(X[:N_train]), np.array(X[N_train:N_train+N_val]), np.array(X[N_train+N_val:])
    # Y_train, Y_val, Y_test = np.array(Y[:N_train]), np.array(Y[N_train:N_train+N_val]), np.array(Y[N_train+N_val:])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=N_test, random_state=SEED)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=N_val, random_state=SEED)

    """
    모델 설정
    """
    n_in = len(X[0][0])
    n_out = len(Y[0])
    n_hidden = [20, 20, 20, 20, 20]

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    """
    모델 구축
    """
    model = Sequential()

    model.add(LSTM(
        n_hidden[0],
        input_shape=(window, n_in),
        # activation='softsign',
        # kernel_initializer='glorot_normal', # Xavier
        return_sequences=True))
    model.add(Dropout(0.3))

    for i in range(len(n_hidden) - 2):
        model.add(LSTM(
            n_hidden[i + 1],
            # activation='softsign',
            # kernel_initializer='glorot_normal',  # Xavier
            return_sequences=True))
        model.add(Dropout(0.3))

    model.add(LSTM(
        n_hidden[len(n_hidden) - 1],
        # activation='softsign',
        # kernel_initializer='glorot_normal',  # Xavier
        return_sequences=False))

    model.add(Dense(
        n_out,
        activation='linear'))

    model.summary()

    optimizer = adam(lr=0.005)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    """
    모델 학습
    """
    history = model.fit(X_train, Y_train,
                        validation_data=(X_val, Y_val),
                        epochs=200,
                        callbacks=[early_stopping],
                        verbose=2)

    """
    Loss
    """
    plt.plot(history.history['loss'])
    plt.title('Loss')
    plt.show()

    """
    학습 후 출력
    """
    output = scaler.inverse_transform(model.predict(X))
    target = scaler.inverse_transform(Y)
    plt.plot(output[::, 3], 'b--', label='output')
    plt.plot(target[::, 3], 'r-', label='target')
    plt.legend()
    plt.title('After Training - close')
    plt.show()

    """
    모델 평가
    """
    output = scaler.inverse_transform(model.predict(X_test))
    target = scaler.inverse_transform(Y_test)
    score = sqrt(mean_squared_error(target[::, 3], output[::, 3]))
    print('Train Score: %.2f RMSE' % score)

    """
    다음 일주일 예측
    """
