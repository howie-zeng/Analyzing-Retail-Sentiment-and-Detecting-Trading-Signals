import os
from abc import ABC, abstractmethod
from collections import defaultdict
import warnings

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import helper 
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseModel(ABC):

    def __init__(self):
        super().__init__()
        self.model = None
        self.feature_importances = defaultdict(float)

class XGBoost(BaseModel):

    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn 

    def train(self, X, y):
        self.model = xgb.XGBRegressor(objective=self.loss_fn, n_jobs=-1, random_state=helper.RANDOM_STATE)
        self.model.fit(X, y)

        for i, col in enumerate(X.columns):
            self.feature_importances[col] += self.model.feature_importances_[i]

    def get_feature_importance(self):
        return self.feature_importances

    def predict(self, X):
        return self.model.predict(X)
class RandomForest(BaseModel):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn
        self.feature_importances = {} 

    def train(self, X, y):
        self.model = RandomForestRegressor(criterion=self.loss_fn, n_jobs=-1, random_state=helper.RANDOM_STATE)
        self.model.fit(X, y)

        for col, importance in zip(X.columns, self.model.feature_importances_):
            self.feature_importances[col] = self.feature_importances.get(col, 0) + importance

    def get_feature_importance(self):
        return self.feature_importances

    def predict(self, X):
        return self.model.predict(X)
    

class StockPredictor:

    def __init__(self, model, window_size=200, stationary=False):
        self.window_size = window_size
        self.model = model
        self.prepare_data()

    def prepare_data(self):
        df = self.stock_data[self.stock].copy()
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 
                 'MA5', 'MA10', 'MA20', 'MA50', 'MA200', 
                 'WVAD', 'MACD',  'RSI', 'macd_line', 'signal_line', 'CCI', 
                 'BB_Upper', 'BB_Lower', 'Buy_Signal', 'Sell_Signal', 
                 'WVF_color', 'WVF', 'upperBand', 'rangeHigh',
                 'VPT', 'AD'
                ]] 

        # Merging other stocks, economic indicator
        dataToAdd = ['^TNX', 'SPY', 'QQQ', 'DIA'] # 'QQQ', "SPY", "MSFT", "AMZN", "GOOG",'DIA'
        for item in dataToAdd:
            df = df.merge(self.stock_data[item][['Date', 'Close']], on="Date", how='inner', suffixes=("", f'_{item}'))

        df.set_index('Date', inplace=True)
        df = df.loc[self.fromDate:self.toDate]
        df[f'Close_{self.lag}_days_later'] = df['Close'].shift(-self.lag)

        # Uncomment if you want to introduce these features later
        # df['relative_volume'] = df['Volume'] / self.data['MA_Volume']
        # df['rsi_class'] = df['rsi_class'].astype('category')
        # df['rsi_class'] = df['rsi_class'].cat.codes
        # df['greenDay'] = df['Open'] > self.data['Close']
        # df['open_close_diff'] = df['Close'] - self.data['Open']/ self.data['Open']
        # df['high_close_diff'] = df.apply(lambda row: (row['High'] - row['Low'])/row['Low'] * (1 if row['greenDay'] else -1), axis=1)
        # df['greenDay'] = df['Open'] > df['Close']
        # df['open_close_diff'] = (df['Close'] - df['Open']) / df['Open']
        # df['high_close_diff'] = df.apply(lambda row: (row['High'] - row['Low']) / row['Low'] * (1 if row['greenDay'] else -1), axis=1)
        
        df['MACD'] = df['macd_line'] - df['signal_line']
        for i in range(1, 11):  # range - 1 lag days
            df[f'close_lag_{i}'] = df['Close'].shift(i)
            #df[f'close_spy_lag_{i}'] = df['Close_SPY'].shift(i)
            #df[f'close_qqq_lag_{i}'] = df['Close_QQQ'].shift(i)
            
            #df[f'volume_lag_{i}'] = df['Volume'].shift(i)
            # df[f'Open_lag_{i}'] = df['Open'].shift(i)
            # df[f'High_lag_{i}'] = df['High'].shift(i)
            # df[f'Low_lag_{i}'] = df['Low'].shift(i)
            # df[f'volume_lag_{i}'] = df['Volume'].shift(i)
            # df[f'RSI_lag_{i}'] = df['RSI'].shift(i)

        df.dropna(inplace=True)

        toDrop = ['Open', 'High', 'Low', 'Close_{}_days_later'.format(self.lag)]
        self.df = df
        self.X = df.drop(toDrop, axis=1, errors='ignore')
        self.y = df[f'Close_{self.lag}_days_later']


    def fit_predict(self):
        predictions = []
        true_values = []

        for start in tqdm(range(0, X.shape[0] - self.window_size), desc="Processing"):
            end = start + self.window_size
            
            X_train, X_test = X.iloc[start:end], X.iloc[end:end+1]
            y_train, y_test = y.iloc[start:end], y.iloc[end:end+1]

            self.model.train(X_train, y_train)
            y_pred = self.model.predict(X_test)
            predictions.append(y_pred[0])
            true_values.append(y_test.iloc[0])

        self.predictions = predictions
        self.true_values=true_values
        self.current_prices=current_prices
    
    def plot_residuals(self):
        true_returns, predicted_returns = self.true_returns, self.predicted_returns
        residuals = [(pred - true) for true, pred in zip(true_returns, predicted_returns)]
        plot_dates = self.dates[self.window_size:]
        plt.figure(figsize=(12, 6))
        plt.scatter(plot_dates, residuals, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.title('Residuals Analysis')
        plt.ylabel('Residuals')
        plt.xlabel('Date')
        plt.xticks(rotation=45)  
        plt.tight_layout() 
        plt.show()

    def show_feature_importance(self, n=20):
        feature_importances = self.model.get_feature_importance()

        sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:n]
        features, importance_values = zip(*sorted_features)

        plt.figure(figsize=(16, 8))
        plt.barh(features, importance_values)
        plt.xlabel('Importance')
        plt.title(f'Top {n} Average Feature Importances Across All Models')
        plt.gca().invert_yaxis()  
        plt.show()




    