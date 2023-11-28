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

    def __init__(self, loss_fn, params = {}):
        super().__init__()
        self.loss_fn = loss_fn 
        self.params = params

    def train(self, X, y):
        params = {
            #'tree_method': 'hist',
            #'device': device,  # Use GPU
            **self.params,
            'objective': self.loss_fn,
            'n_jobs': -1,
            'random_state': helper.RANDOM_STATE
        }
        self.model = xgb.XGBRegressor(**params)
        
        self.model.fit(X, y)

        for i, col in enumerate(X.columns):
            self.feature_importances[col] += self.model.feature_importances_[i]

    def get_feature_importance(self):
        return self.feature_importances

    def predict(self, X):
        return self.model.predict(X)
class RandomForest(BaseModel):
    def __init__(self, loss_fn, params={}):
        super().__init__()
        self.loss_fn = loss_fn
        self.params = params
        self.feature_importances = {} 

    def train(self, X, y):
        params = {
            **self.params,
            'criterion': self.loss_fn, 
            'n_jobs': -1, 
            'random_state': helper.RANDOM_STATE
             }
        self.model = RandomForestRegressor(**params)
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
        self.stationary = stationary

    def fit_predict(self, X, y, df_stock):
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
        self.true_values = true_values
        self.X = X
        self.y = y
        self.dates = X.index.values

        # Calculate returns if stationary
        if not self.stationary:
            close_prices = df_stock['Close'].values[self.window_size:]
            self.true_returns = (np.array(true_values) - close_prices) / close_prices * 100
            self.predicted_returns = (np.array(predictions) - close_prices) / close_prices * 100
        else:
            self.true_returns = self.true_values
            self.predicted_returns = self.predictions

    def print_metrics(self):
        true_returns, predicted_returns = self.true_returns, self.predicted_returns
        MSE = mean_squared_error(true_returns, predicted_returns)
        MAE = mean_absolute_error(true_returns, predicted_returns)

        print(f"Mean Squared Error (MSE): {MSE:.4f}")
        print(f"Mean Absolute Error (MAE): {MAE:.4f}")

        self.MSE = MSE
        self.MAE = MAE

        return MAE

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
        plt.grid()
        plt.show()




    