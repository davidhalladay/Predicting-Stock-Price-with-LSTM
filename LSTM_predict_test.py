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

def model(b_size):
    model_lstm = Sequential()
    model_lstm.add(LSTM(16, input_shape=(1, b_size), activation='relu', return_sequences=False))
    model_lstm.add(Dense(1))
    model_lstm.compile(loss='mean_squared_error', optimizer='adam')
    return model_lstm

def train(stock_name):
    # loading data
    raw_data = pd.read_csv("./data/%s.csv"%(stock_name),na_values=['null'],index_col='Date',parse_dates=True,infer_datetime_format=True).astype(float)
    feature_cols = ['Open', 'High', 'Low', 'Volume']
    data_adj_close = pd.DataFrame(raw_data['Adj Close'])
    normalized_data = normalize(raw_data[feature_cols],feature_cols,valid_show = valid_mode)

    print("<"+"="*50+">")
    print("Total number of data : ",raw_data.shape)
    data_adj_close_last3day = data_adj_close[-3:]

    data_X = normalized_data[:-3]
    print("Data X :",data_X.shape , "last 3 days Data Y :",data_adj_close_last3day.shape)
    print("<"+"="*50+">")

    test_X =np.array(data_X).reshape(data_X.shape[0], 1, data_X.shape[1])

    LSTM_model_1d = model(test_X.shape[2])
    print("<"+"="*50+">")
    print("Loading data...")
    checkpoint_path = "./save_model/day1.ckpt"
    LSTM_model_1d.load_weights(checkpoint_path)

    LSTM_model_2d = model(test_X.shape[2])
    checkpoint_path = "./save_model/day2.ckpt"
    LSTM_model_2d.load_weights(checkpoint_path)

    LSTM_model_3d = model(test_X.shape[2])
    checkpoint_path = "./save_model/day3.ckpt"
    LSTM_model_3d.load_weights(checkpoint_path)

    # predict train data
    pred_test_1d = LSTM_model_1d.predict(test_X)
    pred_test_1d = pred_test_1d[-1] - pred_test_1d[-2]
    pred_test_2d = LSTM_model_2d.predict(test_X)
    pred_test_2d = pred_test_2d[-1] - pred_test_2d[-3]
    pred_test_3d = LSTM_model_3d.predict(test_X)
    pred_test_3d = pred_test_3d[-1] - pred_test_3d[-4]

    # predict valid data
    pred_test_last3day = np.array([pred_test_1d,pred_test_2d,pred_test_3d]).reshape(-1)
    pred_test_last3day = pred_test_last3day + data_adj_close.values[-3]
    print("Today price :",data_adj_close.values.reshape(-1)[-4])
    print("Last 3 day pred price :",pred_test_last3day)
    print("Last 3 day True price :",data_adj_close_last3day.values.reshape(-1))
    #print("R2 score (Test data) :\t%.4f"%(r2_score(pred_test_last3day, data_adj_close_last3day)))

    # loss
    today = data_adj_close.values.reshape(-1)[-4]
    pred_test = pred_test_last3day
    true_test = data_adj_close_last3day.values.reshape(-1)
    diff = abs(pred_test - true_test)
    cc_pred_test = pred_test - today
    cc_true_test = true_test - today
    correction = (cc_pred_test * cc_true_test) >= 0

    # plot this week
    true_week = data_adj_close.values.reshape(-1)[-7:]
    tmp_week = data_adj_close.values.reshape(-1)[-7:-3]
    pred_week = np.concatenate((tmp_week,pred_test_last3day),0)
    print(true_week)
    print(pred_week)
    plt.figure(figsize = (10,8))
    plt.plot(true_week, label='Ground True',color = 'r')
    plt.plot(pred_week, label='LSTM pred',color = 'b')
    plt.title("LSTM Prediction (%d day)" %(7))
    plt.gca().legend(('true_week','pred_week'))
    plt.xlabel('Day (on time series)')
    plt.ylabel('Price (USD)')
    for a, b in zip([0,1,2,3,4,5,6], true_week):
        plt.text(a, b, "%.4f"%(b), va='bottom', fontsize=8)
    for a, b in zip([0,1,2,3,4,5,6], pred_week):
        plt.text(a, b, "%.4f"%(b), va='bottom', fontsize=8)
    #plt.legend()
    plt.savefig("./output/%s_weekly.png" %(stock_name))

    return diff , correction


def analysis():
    stock_path = "./data"
    stock_name = os.listdir(stock_path)
    loss_1d = 0.
    loss_2d = 0.
    loss_3d = 0.
    loss_1d_list = []
    loss_2d_list = []
    loss_3d_list = []
    correct_1d = 0.
    correct_2d = 0.
    correct_3d = 0.
    correct_1d_list = []
    correct_2d_list = []
    correct_3d_list = []
    for stock in stock_name:
        stock = stock.replace(".csv", "")
        loss ,correct = train(stock)
        loss_1d += loss[0]
        loss_2d += loss[1]
        loss_3d += loss[2]
        if correct[0] == False : correct_1d += 1.
        if correct[1] == False : correct_2d += 1.
        if correct[2] == False : correct_3d += 1.
        loss_1d_list.append(loss_1d)
        loss_2d_list.append(loss_2d)
        loss_3d_list.append(loss_3d)
        correct_1d_list.append(correct_1d)
        correct_2d_list.append(correct_2d)
        correct_3d_list.append(correct_3d)

    print("Average loss 1 day : ",loss_1d/len(stock_name))
    print("Average loss 2 day : ",loss_2d/len(stock_name))
    print("Average loss 3 day : ",loss_3d/len(stock_name))
    print("Probability loss 1 day : ",correct_1d/len(stock_name))
    print("Probability loss 2 day : ",correct_2d/len(stock_name))
    print("Probability loss 3 day : ",correct_3d/len(stock_name))

    plt.figure(figsize = (10,8))
    plt.plot(loss_1d_list, label='1 day loss')
    plt.plot(loss_2d_list, label='2 day loss')
    plt.plot(loss_3d_list, label='3 day loss')
    plt.gca().legend(('1 day','2 day',"3 day"))
    plt.title("Prediction loss")
    plt.xlabel('number of stocks')
    plt.ylabel('loss (square error)')
    #plt.legend()
    plt.savefig("./output/loss_analysis.png" )

    plt.figure(figsize = (10,8))
    plt.plot(correct_1d_list, label='Probability loss 1 day')
    plt.plot(correct_2d_list, label='Probability loss 2 day')
    plt.plot(correct_3d_list, label='Probability loss 3 day')
    plt.gca().legend(('1 day','2 day',"3 day"))
    plt.title("Prediction loss")
    plt.xlabel('number of stocks')
    plt.ylabel('Probability loss')
    plt.savefig("./output/Probability_analysis.png" )


def AMD_analysis():
    stock_path = "./AMD_data"
    stock_name = os.listdir(stock_path)
    loss_1d = 0.
    loss_2d = 0.
    loss_3d = 0.
    loss_1d_list = []
    loss_2d_list = []
    loss_3d_list = []
    correct_1d = 0.
    correct_2d = 0.
    correct_3d = 0.
    correct_1d_list = []
    correct_2d_list = []
    correct_3d_list = []
    for stock in stock_name:
        stock = stock.replace(".csv", "")
        loss ,correct = train(stock)
        loss_1d += loss[0]
        loss_2d += loss[1]
        loss_3d += loss[2]
        if correct[0] == False : correct_1d += 1.
        if correct[1] == False : correct_2d += 1.
        if correct[2] == False : correct_3d += 1.
        loss_1d_list.append(loss_1d)
        loss_2d_list.append(loss_2d)
        loss_3d_list.append(loss_3d)
        correct_1d_list.append(correct_1d)
        correct_2d_list.append(correct_2d)
        correct_3d_list.append(correct_3d)

    print("Average loss 1 day : ",loss_1d/len(stock_name))
    print("Average loss 2 day : ",loss_2d/len(stock_name))
    print("Average loss 3 day : ",loss_3d/len(stock_name))
    print("Probability loss 1 day : ",correct_1d/len(stock_name))
    print("Probability loss 2 day : ",correct_2d/len(stock_name))
    print("Probability loss 3 day : ",correct_3d/len(stock_name))

    plt.figure(figsize = (10,8))
    plt.plot(loss_1d_list, label='1 day loss')
    plt.plot(loss_2d_list, label='2 day loss')
    plt.plot(loss_3d_list, label='3 day loss')
    plt.gca().legend(('1 day','2 day',"3 day"))
    plt.title("Prediction loss")
    plt.xlabel('number of stocks')
    plt.ylabel('loss (square error)')
    #plt.legend()
    plt.savefig("./output/AMD_loss_analysis.png" )

    plt.figure(figsize = (10,8))
    plt.plot(correct_1d_list, label='Probability loss 1 day')
    plt.plot(correct_2d_list, label='Probability loss 2 day')
    plt.plot(correct_3d_list, label='Probability loss 3 day')
    plt.gca().legend(('1 day','2 day',"3 day"))
    plt.title("Prediction loss")
    plt.xlabel('number of stocks')
    plt.ylabel('Probability loss')
    plt.savefig("./output/AMD_Probability_analysis.png" )

if __name__ == '__main__':
    analysis()
