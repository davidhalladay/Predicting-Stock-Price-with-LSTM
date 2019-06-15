import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM,Dense,Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.dates as mdates
from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import LSTM
import types
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


valid_mode = False
valid_size = 100
num_epoch = 100
batch_size = 8

def my_plot(valid_Y,pred_valid,day = "1",filename = ""):
    plt.figure(figsize = (10,8))
    plt.plot(valid_Y, label='Ground True')
    plt.plot(pred_valid, label='LSTM pred')
    plt.title("LSTM Prediction (%s day)" %(day))
    plt.xlabel('Day (on time series)')
    plt.ylabel('Price (USD)')
    #plt.legend()
    plt.savefig("./output/%s.png" %(filename))

    return 0

def normalize(data,feature_cols,valid_show = False):
    # normalize to ( 0 ~ 1 )
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    normalized_data = pd.DataFrame(columns=feature_cols, data=normalized_data, index=data.index)
    if valid_show == True:
        print('Shape of features : ', feature_minmax_transform.shape)
        print(normalized_data.head())
    return normalized_data

def score_show(X,Y,model, model_name,show_img = False):
    pred_X = model.predict(X)
    RMSE_score = np.sqrt(mean_squared_error(Y, pred_X))
    print('RMSE: ', RMSE_score)
    R2_score = r2_score(Y, pred_X)
    print('R2 score: ', R2_score)

    if show_img == True:
        plt.plot(validation_y.index, predicted,'r', label='Predict')
        plt.plot(validation_y.index, validation_y,'b', label='Actual')
        plt.ylabel('Price')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.title(model_name + ' Predict vs Actual')
        plt.legend(loc='upper right')
        plt.show()
    return 0

def model(b_size):
    model_lstm = Sequential()
    model_lstm.add(LSTM(16, input_shape=(1, b_size), activation='relu', return_sequences=False))
    model_lstm.add(Dense(1))
    model_lstm.compile(loss='mean_squared_error', optimizer='adam')
    return model_lstm

def train():
    # loading data
    raw_data = pd.read_csv("./data/AMD_full.csv",na_values=['null'],index_col='Date',parse_dates=True,infer_datetime_format=True).astype(float)
    feature_cols = ['Open', 'High', 'Low', 'Volume']
    data_adj_close = pd.DataFrame(raw_data['Adj Close'])
    normalized_data = normalize(raw_data[feature_cols],feature_cols,valid_show = valid_mode)

    print("<"+"="*50+">")
    print("Total number of data : ",raw_data.shape)
    data_adj_close_1day = data_adj_close.shift(-1)
    data_adj_close_2day = data_adj_close.shift(-2)
    data_adj_close_3day = data_adj_close.shift(-3)

    data_X = normalized_data[:-1]
    data_Y_1d = data_adj_close_1day[:-1]
    data_Y_2d = data_adj_close_2day[:-2]
    data_Y_3d = data_adj_close_3day[:-3]
    print("Data X :",data_X.shape , "Data Y :",data_Y_1d.shape)
    print("<"+"="*50+">")
    print("TimeSeriesSplit...")
    timeSeries_split= TimeSeriesSplit(n_splits=10)
    for train_index, valid_index in timeSeries_split.split(data_X):
            train_X, valid_X = data_X[:len(train_index)], data_X[len(train_index): (len(train_index)+len(valid_index))]
            train_Y_1d, valid_Y_1d = data_Y_1d[:len(train_index)].values.ravel(), data_Y_1d[len(train_index): (len(train_index)+len(valid_index))].values.ravel()
            train_Y_2d, valid_Y_2d = data_Y_2d[:len(train_index)].values.ravel(), data_Y_2d[len(train_index): (len(train_index)+len(valid_index))].values.ravel()
            train_Y_3d, valid_Y_3d = data_Y_3d[:len(train_index)].values.ravel(), data_Y_3d[len(train_index): (len(train_index)+len(valid_index))].values.ravel()
    print("train_X.shape : ",train_X.shape)
    print("valid_X.shape : ",valid_X.shape)
    print("train_Y_1d.shape : ",train_Y_1d.shape)
    print("valid_Y_1d.shape : ",valid_Y_1d.shape)
    print("train_Y_2d.shape : ",train_Y_2d.shape)
    print("valid_Y_2d.shape : ",valid_Y_2d.shape)
    print("train_Y_3d.shape : ",train_Y_3d.shape)
    print("valid_Y_3d.shape : ",valid_Y_3d.shape)

    # DecisionTreeRegressor valid
    # DT_Reg = DecisionTreeRegressor(random_state=312)
    # DTR_clf = DT_Reg.fit(train_X, train_Y)
    # score_show(train_X,train_Y,DTR_clf, 'Decision Tree Regression')

    train_X =np.array(train_X).reshape(train_X.shape[0], 1, train_X.shape[1])
    valid_X =np.array(valid_X).reshape(valid_X.shape[0], 1, valid_X.shape[1])

    early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
    K.clear_session()
    LSTM_model_1d = model(train_X.shape[2])
    print("<"+"="*50+">")
    print("Train model predict first day...")
    print(train_X.shape)
    print(train_Y_1d.shape)
    LSTM_model_1d.fit(train_X, train_Y_1d, epochs=num_epoch, batch_size=batch_size, verbose=1, shuffle=False, callbacks=[early_stop])

    LSTM_model_2d = model(train_X.shape[2])
    print("<"+"="*50+">")
    print("Train model predict second day...")
    LSTM_model_2d.fit(train_X, train_Y_2d, epochs=num_epoch, batch_size=batch_size, verbose=1, shuffle=False, callbacks=[early_stop])

    LSTM_model_3d = model(train_X.shape[2])
    print("<"+"="*50+">")
    print("Train model predict third day...")
    LSTM_model_3d.fit(train_X, train_Y_3d, epochs=num_epoch, batch_size=batch_size, verbose=1, shuffle=False, callbacks=[early_stop])

    # predict train data
    pred_train_1d = LSTM_model_1d.predict(train_X)
    pred_train_2d = LSTM_model_2d.predict(train_X)
    pred_train_3d = LSTM_model_3d.predict(train_X)

    # predict valid data
    pred_valid_1d = LSTM_model_1d.predict(valid_X)
    pred_valid_2d = LSTM_model_2d.predict(valid_X)
    pred_valid_3d = LSTM_model_3d.predict(valid_X)
    print("R2 score (Train data) :\t%.4f\t%.4f\t%.4f"%(r2_score(train_Y_1d, pred_train_1d),r2_score(train_Y_2d, pred_train_2d),r2_score(train_Y_3d, pred_train_3d)))
    print("R2 score (valid data) :\t%.4f\t%.4f\t%.4f"%(r2_score(valid_Y_1d, pred_valid_1d),r2_score(valid_Y_2d, pred_valid_2d[:-1]),r2_score(valid_Y_3d, pred_valid_3d[:-2])))

    print("<"+"="*50+">")
    LSTM_score_1d = LSTM_model_1d.evaluate(valid_X, valid_Y_1d, batch_size=1)
    LSTM_score_2d = LSTM_model_2d.evaluate(valid_X[:-1], valid_Y_2d, batch_size=1)
    LSTM_score_3d = LSTM_model_3d.evaluate(valid_X[:-2], valid_Y_3d, batch_size=1)
    print('LSTM score: %.4f\t%.4f\t%.4f' %(LSTM_score_1d,LSTM_score_2d,LSTM_score_3d))
    pred_valid_1d = LSTM_model_1d.predict(valid_X)
    pred_valid_2d = LSTM_model_2d.predict(valid_X)
    pred_valid_3d = LSTM_model_3d.predict(valid_X)

    # plot
    my_plot(valid_Y_1d,pred_valid_1d,day = "1",filename = "LSTM_Prediction_1d")
    my_plot(valid_Y_2d,pred_valid_2d[:-1],day = "2",filename = "LSTM_Prediction_2d")
    my_plot(valid_Y_3d,pred_valid_3d[:-2],day = "3",filename = "LSTM_Prediction_3d")

    return 0


if __name__ == '__main__':
    train()
