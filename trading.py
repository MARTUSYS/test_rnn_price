from alpha_vantage.timeseries import TimeSeries

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tqdm import tqdm

import datetime

from sklearn.preprocessing import MinMaxScaler


def to_datetime(df):
    date = datetime.datetime.strptime(str(df).split(' ')[0], '%Y-%m-%d')
    return date.strftime("%Y-%m-%d")


class data_price():
    def __init__(self, name, key, num_shape_test=360, window=60, date_start=None, path=None):

        self.key = key

        self.name = name
        self.num_shape_test = num_shape_test
        self.window = window
        self.date_start = date_start

        self.data = None
        self.path = path

        self.train = None
        self.x = None
        self.y = None

        self.test = None
        self.X_test = None

        self.df_volume = None

        self.sc = MinMaxScaler(feature_range=(0, 1))

        self.get_price()
        self.transformation_data(self.window)

    def get_price(self):
        """
        Получение данных и разделение на train и test
        :return:
        """
        if self.path:

            self.data = pd.read_csv(self.path)
            self.data = self.data.iloc[::-1]

            # Разделение для временного ряда на train и test только по столбцу "цена закрытия"
            self.train = self.data.iloc[:-self.num_shape_test, 1:].values
            self.test = self.data.iloc[-self.num_shape_test:, 1:].values

            '''ticker = 'baba'
            adjusted = 'true'

            for i in [1, 5, 15, 30, 60]:
                time.sleep(15)
                interval = f'{i}min'
                print(interval)

                for year in range(1, 3):
                    for month in range(1, 13):
                        print(f'year: {year}\tmonth: {month}')
                        date = f'year{year}month{month}'

                        csv_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={ticker}&interval={interval}&slice={date}&adjusted={adjusted}&apikey={key}&datatype=csv&outputsize=full'

                        if year == 1 and month == 1:
                            df = pd.read_csv(csv_url)

                        time.sleep(15)
                        df = pd.concat([df, pd.read_csv(csv_url)])

                df.to_csv(f'{ticker}-{interval}.csv', index=False, header=True)
                df = pd.read_csv(f'{ticker}-{interval}.csv')'''
        else:
            ts = TimeSeries(key=self.key, output_format='pandas')  # numpy
            self.data, meta_data = ts.get_daily(symbol=self.name, outputsize='full')
            # self.data = self.data.index.sort_values.reset_index(drop=True)


            # ti = TechIndicators(key=key, output_format='pandas')
            # data_RSI, meta_data = ti.get_rsi(symbol=self.name, interval='daily', time_period=60, series_type='close')

            # Извлечение даты из индекса
            # self.data['4. close'] = self.data['4. close'].astype(float)
            self.data['Date'] = self.data.index.values
            self.data['Date'] = self.data['Date'].apply(lambda x: to_datetime(x))

            # Фильтр по дате ("%Y-%m-%d")
            if self.date_start:
                self.data = self.data[(self.data['Date'] > self.date_start)]

            self.data = self.data.iloc[::-1]

            # Разделение для временного ряда на train и test только по столбцу "цена закрытия"
            self.train = self.data.iloc[:-self.num_shape_test, 0:4].values
            self.test = self.data.iloc[-self.num_shape_test:, 0:4].values

    def transformation_data(self, window):
        """
        Разделяет на X и Y и преобразует данные, подготавливает тест
        :return:
        """
        self.window = window
        # Преобразование для обучения
        train_scaled = self.sc.fit_transform(self.train)

        self.x = []
        self.y = []  # Price on next day

        for i in range(window, train_scaled.shape[0]):
            X_train_ = np.reshape(train_scaled[i - window:i, 0:train_scaled.shape[-1]], (window, train_scaled.shape[-1]))  ##
            self.x.append(X_train_)
            self.y.append(train_scaled[i, :])  ##
        self.x = np.stack(self.x)
        self.y = np.stack(self.y)
        print('x:', self.x.shape)
        print('y:', self.y.shape)

        # Преобразование для теста
        self.df_volume = np.vstack((self.train, self.test))

        inputs = self.df_volume[self.df_volume.shape[0] - self.test.shape[0] - window:]
        # inputs = inputs.reshape(-1, 1)  ##
        inputs = self.sc.transform(inputs)

        num_2 = self.num_shape_test + window
        X_test = []
        for i in range(window, num_2):
            X_test_ = np.reshape(inputs[i - window:i, 0:inputs.shape[-1]], (window, inputs.shape[-1]))  ##
            X_test.append(X_test_)
        self.X_test = np.stack(X_test)
        print("X_test:", self.X_test.shape)

    def save(self, filename=''):
        np.savez_compressed(f'{self.name}_{filename}', self.train, self.test)

    def load(self, filename=''):
        self.train = np.load(f'{self.name}_{filename}.npz')['arr_0']
        self.test = np.load(f'{self.name}_{filename}.npz')['arr_1']

    def get_data_from_predict(self):
        """
        Данные для предсказания
        :return: Последние данные (1, window) размера
        """
        data_f_pr = self.test[-self.window:]
        data_f_pr = self.sc.transform(data_f_pr)
        data_f_pr = data_f_pr.reshape(1, self.window, data_f_pr.shape[-1])  ###
        return data_f_pr

    def get_training_data(self):
        return [self.x, self.y]

    def get_test_data(self):
        return self.X_test, self.sc.transform(self.test)

    def get_inverse_data(self, data):
        return self.sc.inverse_transform(data)

    def grafica(self):
        plt.figure(figsize=(20, 7))
        if self.path:
            plt.plot(self.data['time'], self.data['close'].values, label=f'{self.name} Stock Price', color='red')
        else:
            plt.plot(self.data['Date'], self.data['4. close'].values, label=f'{self.name} Stock Price', color='red')
        plt.xticks(np.arange(100, self.data.shape[0], 200))
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.show()

    def metrics(self, predict, k=0.9):
        print('open   high  low  close')
        starting_date = round(self.data.shape[0] * k)

        predict = self.sc.inverse_transform(predict)
        diff = predict - self.test

        print("MSE:", np.mean(diff ** 2))
        print("MAE:", np.mean(abs(diff)))
        print("RMSE:", np.sqrt(np.mean(diff ** 2)))

        plt.figure(figsize=(20, 7))
        if self.path:
            plt.plot(self.data['time'].values[starting_date:], self.df_volume.iloc[starting_date:, 0:4], color='red',
                     label=f'Real {self.name} Stock Price')
            plt.plot(self.data['time'][-predict.shape[0]:].values, predict, color='blue',
                     label=f'Predicted {self.name} Stock Price')
        else:
            plt.plot(self.data['Date'].values[starting_date:], self.df_volume[starting_date:], color='red',
                     label=f'Real {self.name} Stock Price')
            plt.plot(self.data['Date'][-predict.shape[0]:].values, predict, color='blue',
                     label=f'Predicted {self.name} Stock Price')

        plt.xticks(np.arange(100, self.data[starting_date:].shape[0], 200))
        plt.title(f'{self.name} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.show()


class lstm_model():
    def __init__(self, compile_lr_schedule=True, units=50, dropout=0.2, learning_rate=1e-3, outputsize=4):
        self.model = None
        self.history = None
        self.LEARNING_RATE = learning_rate
        self.DECAY_STEPS = 100
        self.DECAY_RATE = 0.96
        self.compile_lr_schedule = compile_lr_schedule
        self.PATIENCE = 5

        self.model = keras.Sequential([
            layers.Bidirectional(
                layers.LSTM(units=units, dropout=dropout, return_sequences=True)  # input_shape=(X_train.shape[1], 1))
            ),
            layers.Bidirectional(
                keras.layers.LSTM(units=units, dropout=dropout, return_sequences=True)
            ),
            layers.Bidirectional(
                layers.LSTM(units=units, dropout=dropout, return_sequences=True)
            ),

            layers.LSTM(units=units, dropout=dropout),

            layers.Dense(units=outputsize)  ##

        ])
        if self.compile_lr_schedule:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.LEARNING_RATE,
                decay_steps=self.DECAY_STEPS, decay_rate=self.DECAY_RATE,
                staircase=True)
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                                loss=tf.keras.losses.MeanSquaredError(),
                                metrics=[tf.keras.metrics.RootMeanSquaredError()])
        else:
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE), loss='mean_squared_error', metrics=['MAE'])

    def fit_model(self, data_price_class, epochs=100, batch_size=512):
        # Извлечение данных из класса с ценами на акцию
        X = data_price_class.x
        y = data_price_class.y
        val_data = data_price_class.get_test_data()

        # Обучение
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'model.h5',
            monitor='val_loss',
            verbose=1,
            save_best_only=True
        )
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=self.PATIENCE, restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            verbose=1,
            patience=10,
            min_lr=1e-5
        )
        # early_stop / reduce_lr / checkpoint // Остановка обучения / Снижеение скорости обучения / Сохраниение модели
        if self.compile_lr_schedule:
            callbacks = [early_stop, checkpoint]
        else:
            callbacks = [reduce_lr, checkpoint]

        self.history = self.model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            verbose=1,
            shuffle=True,
            callbacks=callbacks
        )

    def load_weights(self, name=''):
        self.model.load_weights(f'{name}model.h5')

    def grafica_loss(self):
        for i, j in enumerate([('loss', 'val_loss'), ('MAE', 'val_MAE')]):
            fig, axes = plt.subplots(figsize=(6, 2), dpi=100)
            axes.set_xlabel('epoch')
            axes.set_ylabel(j[0])
            axes.set_title(j[0])
            axes.plot(self.history.epoch, self.history.history[j[0]])
            axes.plot(self.history.epoch, self.history.history[j[1]])
            axes.legend([f'train_{j[0]}', j[1]])

    def predict_period(self, data_price_class, interval=20):
        window = data_price_class.window
        data_from_predict = data_price_class.get_data_from_predict()

        predict = []

        data_from_predict = data_from_predict.reshape(window, data_from_predict.shape[-1])  ##

        for _ in tqdm(range(interval)):
            predict_ = self.model.predict(data_from_predict[-window:].reshape(1, window, data_from_predict.shape[-1]))  ##
            predict.append(predict_[0, :])  ##
            data_from_predict = np.row_stack((data_from_predict, predict_))

        predict = data_price_class.get_inverse_data(np.array(predict))  ##

        # predict = predict.reshape(predict.shape[-1], predict.shape[0])
        # label = ['open', 'high', 'low', 'close']
        #
        fig, axes = plt.subplots(figsize=(6, 2), dpi=100)
        # for i in range(len(label)):
        axes.plot(predict)

        if data_price_class.path:
            axes.legend(['open', 'high', 'low', 'close', 'volume'])
        else:
            axes.legend(['open', 'high', 'low', 'close'])

        return predict
