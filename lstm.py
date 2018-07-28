import os
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense,Activation,Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

#ワーニングの出るレベルを上げる
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#numpy のワーニングにログを取らない
warnings.filterwarnings("ignore")

#normalise_window→データを正規化するかどうかの判定
def load_data(filename, seq_len, normalise_window):
    f = open(filename, 'rb').read()
    dataLines = f.decode().split('\n')
    data = []

    #とりあえずcloseの列だけ取得
    for i in range(len(dataLines)):
        data.append(dataLines[i].split(',')[0])

    #いくつまでのデータを見るか
    sequence_length = seq_len + 1
    result = []

    #データ数だけループする
    for index in range(len(data) - sequence_length):
        #resultにdataをsequence_lengthずつずらした範囲のデータを格納
        #resultは[len(data),sequence_length]の配列となる
        result.append(data[index: index + sequence_length])

    #正規化する場合
    if normalise_window:
        result = normalise_windows(result)
    #numpyのリストにする
    result = np.array(result)

    #全データの9割の部分の行番号を取得
    row = round(0.9*result.shape[0])
    #int(row)までの行、列はすべて取得
    train = result[:int(row),:]
    np.random.shuffle(train)

    #訓練データと検証データの分割
    #x_trainにshuffleした株価の変動前の値を格納(N,1)の行列
    x_train = train[:,:-1]
    #y_trainにshuffleした株価の変動後の値を格納（N)のリスト
    y_train = train[:,-1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    #3次元配列に戻す
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [x_train,y_train,x_test,y_test]

#正規化処理
#(pi/p0) - 1
#50個ずつの最初のデータを基準に正規化している
def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def build_model(layers):
    model = Sequential()

    model.add(LSTM(input_shape = (layers[1], layers[0]),
                    output_dim=layers[1],
                    return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(layers[2],return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer='adam')
    print(" 実行時間：　", time.time() - start)
    return model

#現在の株価を渡して次の日の株価を予想する
def predict_point_by_point(model,data):
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

#期間内の株価を渡して次の日の株価を予想する
def predict_sequence_full(model,data,window):
    curr_frame = data[0]
    predicted = []
    for i in xrange(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis = 0)
    return predicted

#複数の株価を渡して明日以降の株価を複数予想する
def predict_sequences_multiple(model, data, window_size, prediction_len):
    prediction_seqs = []
    #50個分のデータを渡して処理を行うため、データ数/50回数処理する
    for i in range(int(len(data)/prediction_len)):
        #50個分のデータを渡す
        curr_frame = data[i * prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:]))
            #predicted.append(model.predict(curr_frame[curr_frame.shape[0],1,curr_frame.shape[1]]))
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 1],predicted[-1],axis = 0)
        prediction_seqs.append(predicted)
    return prediction_seqs
