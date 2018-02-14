import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import GRU
from keras.utils import np_utils
import matplotlib.pyplot as plt


def load_data(window):
    """
    데이터 읽기
    """
    PATH = './result/'
    # open, high, low, close, volume
    raw = np.loadtxt(PATH + 'output.csv', delimiter=',', skiprows=1, usecols=[4, 5])  # select close, volume

    """
    데이터 정규화
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    nptf = scaler.fit_transform(raw)

    """
    데이터 생성
    """
    seq_len = len(raw)  # 전체 시계열 길이
    dim = 2  # 데이터 차원(close, volume)

    X = []
    Y = []  # up: 0, down: 1
    y_prev = nptf[6][0]  # initialization

    for i in range(seq_len - window):
        x_ = nptf[i: i + window]
        X.append(x_)

        y_ = nptf[i + window][0]
        if y_ > y_prev:
            Y.append(0)
        else:
            Y.append(1)

        y_prev = y_

    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    dummy_Y = np_utils.to_categorical(encoded_Y)  # one hot encoded

    X = np.array(X)
    Y = dummy_Y

    return X, Y


def modeling(window, n_in, n_hidden, n_out):
    """
    모델 구축
    """
    model = Sequential()

    model.add(GRU(
        n_hidden[0],
        input_shape=(window, n_in),
        activation='softsign',
        # dropout=0.5,
        return_sequences=True))
    # model.add(Dropout(0.5))

    for i in range(len(n_hidden) - 2):
        model.add(GRU(
            n_hidden[i + 1],
            activation='softsign',
            # dropout=0.5,
            return_sequences=True))
        # model.add(Dropout(0.5))

    model.add(GRU(
        n_hidden[len(n_hidden) - 1],
        activation='softsign',
        return_sequences=False))

    model.add(Dense(
        n_out,
        activation='softmax'))

    return model


def load_weight(model):
    """
    가중치 load
    """
    IN_PATH = './model/'
    try:
        model.load_weights(IN_PATH + 'gru_weights.h5')
        print('Load: gru_weights.h5')
    except:
        print('New model')

    return model


def save_weight(model):
    """
    가중치 save
    """
    OUT_PATH = './model/'
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    model.save_weights(OUT_PATH + 'gru_weights.h5')
    print('Save: gru_weights.h5')


def plot_hist(hist):
    """
    Accuracy & Loss
    """
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='loss')
    acc_ax.plot(hist.history['acc'], 'b', label='acc')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuracy')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')
    plt.show()


# main
if __name__ == "__main__":
    window = 7  # 하나의 시계열 길이

    """
    데이터 로드
    """
    X, Y = load_data(window)

    N_test = int(len(Y) * 0.2)  # split to train and testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=N_test)

    """
    모델 구축
    """
    n_in = len(X[0][0])
    n_hidden = [20, 20, 20, 20, 20]
    n_out = len(Y[0])

    model = modeling(window, n_in, n_hidden, n_out)  # 모델 구축
    model.summary()

    # regression: linear, mean_squared_error
    # classification: softmax, binary_crossentropy
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    """
    모델 학습
    """
    epochs = 100
    units = 100  # save weights per each epochs/units

    for i in range(int(epochs / units)):
        model = load_weight(model)
        hist = model.fit(X_train, Y_train, epochs=units, verbose=2)
        save_weight(model)

    plot_hist(hist)  # draw plot

    """
    모델 평가
    """
    scores = model.evaluate(X_test, Y_test, verbose=2)
    print('\n%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))

    predict = model.predict(X[-1:])  # 예측
    print('UP: %.2f%%, DOWN: %.2f%%' % (predict[0][0], predict[0][1]))  # 확률: [up, down]
