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

class XGBoostModel(BaseModel):

    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn 

    def train(self, X, y):
        self.model = xgb.XGBRegressor(objective=self.loss_fn, n_jobs=-1, random_state=helper.RANDOM_STATE)
        self.model.fit(X, y)

        for i, col in enumerate(X.columns):
            self.feature_importances[col] += self.model.feature_importances_[i]

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

    def predict(self, X):
        return self.model.predict(X)
    

class StockPredictor:

    def __init__(self, df_stock, model, stock, fromDate, toDate, window_size=200, lag=5, stationary=False):
        self.df_stock = df_stock
        self.stock = stock
        self.fromDate = fromDate
        self.toDate = toDate
        self.window_size = window_size
        self.lag = lag
        self.model = model
        self.stationary = stationary


    def fit_predict(self, X, y):
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

    def plot_residuals(self):
        if self.stationary:
            


        residuals = [(pred - true) for true, pred in zip(self.true_values, self.predictions)]
        data = self.df

        plt.figure(figsize=(12, 6))
        plt.scatter(data.index.values[self.window_size:], residuals, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.title('Residuals Analysis')
        plt.ylabel('Residuals %')
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

    
    