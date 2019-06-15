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

test_mode = False
valid_size = 100

def normalize(data,feature_cols,test_show = False):
    # normalize to ( 0 ~ 1 )
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data[feature_cols])
    normalized_data = pd.DataFrame(columns=feature_cols, data=normalized_data, index=data.index)
    if test_show == True:
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
    K.clear_session()
    model_lstm = Sequential()
    model_lstm.add(LSTM(16, input_shape=(1, b_size), activation='relu', return_sequences=False))
    model_lstm.add(Dense(1))
    model_lstm.compile(loss='mean_squared_error', optimizer='adam')

    return model_lstm

def train():
    # loading data
    raw_data = pd.read_csv("./data/AMD.csv",na_values=['null'],index_col='Date',parse_dates=True,infer_datetime_format=True)
    feature_cols = ['Open', 'High', 'Low', 'Volume']
    data_adj_close = pd.DataFrame(raw_data['Adj Close'])
    normalized_data = normalize(raw_data[feature_cols],feature_cols,test_show = test_mode)

    print("<"+"="*50+">")
    print("Total number of data : ",raw_data.shape)
    data_adj_close = data_adj_close.shift(-1)
    valid_X = normalized_data[-valid_size-1:-1]
    valid_Y = data_adj_close[-valid_size-1:-1]
    print("Valid X :",valid_X.shape , "Valid Y :",valid_Y.shape)
    data_X = normalized_data[:-valid_size-1]
    data_Y = data_adj_close[:-valid_size-1]
    print("Data X :",data_X.shape , "Data Y :",data_Y.shape)
    print("<"+"="*50+">")
    print("TimeSeriesSplit...")
    timeSeries_split= TimeSeriesSplit(n_splits=10)
    for train_index, test_index in timeSeries_split.split(data_X):
            train_X, test_X = data_X[:len(train_index)], data_X[len(train_index): (len(train_index)+len(test_index))]
            train_Y, test_Y = data_Y[:len(train_index)].values.ravel(), data_Y[len(train_index): (len(train_index)+len(test_index))].values.ravel()
    print("X_train.shape : ",train_X.shape)
    print("X_test.shape : ",test_X.shape)
    print("y_train.shape : ",train_Y.shape)
    print("y_test.shape : ",test_Y.shape)

    DT_Reg = DecisionTreeRegressor(random_state=312)
    DTR_clf = DT_Reg.fit(train_X, train_Y)
    score_show(train_X,train_Y,DTR_clf, 'Decision Tree Regression')
    train_X =np.array(train_X).reshape(train_X.shape[0], 1, train_X.shape[1])
    test_X =np.array(test_X).reshape(test_X.shape[0], 1, test_X.shape[1])
    print(train_X.shape)
    print(test_X.shape)

    LSTM_model = model(train_X.shape[2])
    early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
    LSTM_model.fit(train_X, train_Y, epochs=200, batch_size=8, verbose=1, shuffle=False, callbacks=[early_stop])

    # predict train data
    pred_train = LSTM_model.predict(train_X)
    # predict test data
    pred_test = LSTM_model.predict(test_X)
    print("R2 score (Train data) :\t%.4f"%(r2_score(train_Y, pred_train)))
    print("R2 score (Test data) :\t%.4f"%(r2_score(test_Y, pred_test)))

    print("<"+"="*50+">")
    LSTM_score = LSTM_model.evaluate(test_X, test_Y, batch_size=1)
    print('LSTM score: %.4f' %(LSTM_score))
    perd_test = LSTM_model.predict(test_X)

    # plot
    plt.figure(figsize = (10,8))
    plt.plot(test_Y, label='Ground True')
    plt.plot(perd_test, label='LSTM pred')
    plt.title("LSTM Prediction")
    plt.xlabel('Observation(on time series)')
    plt.ylabel('INR_Scaled')
    #plt.legend()
    plt.show()

    return 0


if __name__ == '__main__':
    train()
