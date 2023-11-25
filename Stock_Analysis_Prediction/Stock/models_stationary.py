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


class BaseModel(ABC):

    def __init__(self):
        super().__init__()
        self.model = None
        self.feature_importances = defaultdict(float)

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def get_feature_importance(self):
        return self.feature_importances

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

    def __init__(self, stock_data, model, stock='AAPL', fromDate="2015-01-01", toDate='2017-01-01', window_size=200, lag=5):
        self.stock_data = stock_data
        self.stock = stock
        self.fromDate = fromDate
        self.toDate = toDate
        self.window_size = window_size
        self.lag = lag
        self.model = model
        self.prepare_data()

    def prepare_data(self):
        df = self.stock_data[self.stock].copy()
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Close_diff', 'Volume_MA_diff', 
                 'MA5', 'MA10', 'MA20', 'MA50', 'MA200', 
                 'WVAD', 'MACD',  'RSI', 'macd_line', 'signal_line', 'CCI', 
                 'BB_Upper', 'BB_Lower', 'Buy_Signal', 'Sell_Signal', 
                 'WVF_color', 'WVF', 'upperBand', 'rangeHigh'
                 #'VPT', 'AD'
                ]] 

        # Merging other stocks, economic indicator
        dataToAdd = ['^IRX', 'SPY', 'QQQ', 'DIA'] # 'QQQ', "SPY", "MSFT", "AMZN", "GOOG",'DIA'
        for item in dataToAdd:
            if item != "^IRX":
                df = df.merge(self.stock_data[item][['Date', 'Close_diff']], on="Date", how='inner', suffixes=("", f'_{item}'))
            else:
                df = df.merge(self.stock_data[item][['Date', 'Close']], on="Date", how='inner', suffixes=("", f'_{item}'))


        df.set_index('Date', inplace=True)
        df = df.loc[self.fromDate:self.toDate]
        df[f'Return_{self.lag}_days_later'] = df['Close_diff'].shift(-self.lag)

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
            df[f'close_lag_{i}'] = df['Close_diff'].shift(i)        
            df[f'volume_lag_{i}'] = df['Volume_MA_diff'].shift(i)

            #df[f'close_spy_lag_{i}'] = df['Close_SPY'].shift(i)
            #df[f'close_qqq_lag_{i}'] = df['Close_QQQ'].shift(i)
            # df[f'Open_lag_{i}'] = df['Open'].shift(i)
            # df[f'High_lag_{i}'] = df['High'].shift(i)
            # df[f'Low_lag_{i}'] = df['Low'].shift(i)
            # df[f'volume_lag_{i}'] = df['Volume'].shift(i)
            # df[f'RSI_lag_{i}'] = df['RSI'].shift(i)

        df.dropna(inplace=True)

        toDrop = ['Open', 'High', 'Low', 'Return_{}_days_later'.format(self.lag), 'Close']
        self.df = df
        self.X = df.drop(toDrop, axis=1, errors='ignore')
        self.y = df[f'Return_{self.lag}_days_later']


    def fit_predict(self):
        predictions = []
        true_values = []

        for start in tqdm(range(0, self.X.shape[0] - self.window_size), desc="Processing"):
            end = start + self.window_size
            
            X_train, X_test = self.X.iloc[start:end], self.X.iloc[end:end+1]
            y_train, y_test = self.y.iloc[start:end], self.y.iloc[end:end+1]

            self.model.train(X_train, y_train)
            y_pred = self.model.predict(X_test)
            predictions.append(y_pred[0])
            true_values.append(y_test.iloc[0])

        self.predictions = predictions
        self.true_values = true_values
    
    def plot_residuals(self):
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

    def select_top_features(model, X_train, n=10):
        feature_importances = model.get_feature_importance()
        sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:n]
        top_features = [feature for feature, importance in sorted_features]

        X_train_selected = X_train[top_features]

        return X_train_selected, top_features

    def show_feature_importance(self, n=20):
        feature_importances = self.model.get_feature_importance()

        sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:n]
        features, importance_values = zip(*sorted_features)

        # Plotting
        plt.figure(figsize=(16, 8))
        plt.barh(features, importance_values)
        plt.xlabel('Importance')
        plt.title(f'Top {n} Average Feature Importances Across All Models')
        plt.gca().invert_yaxis()  
        plt.show()

    def print_metrics(self):
        MSE = mean_squared_error(self.true_values, self.predictions)
        RMSE = np.sqrt(MSE)
        MAE = mean_absolute_error(self.true_values, self.predictions)

        print(f"Mean Squared Error (MSE): {MSE:.4f}")
        print(f"Root Mean Squared Error (RMSE): {RMSE:.4f}")
        print(f"Mean Absolute Error (MAE): {MAE:.4f}")
        self.MSE = MSE
        self.MAE = MAE
        self.RMSE = RMSE


    def plot_actual_vs_predicted(self):
        prediction_lag = self.lag
        # Define time indices for x-axis based on data length
        time_indices = np.arange(len(self.current_prices))

        # Define indices for plotting the predictions considering the lag
        prediction_indices = time_indices[prediction_lag:]

        # Adjust the predictions list/array to match the adjusted indices
        adjusted_predictions = self.predictions[:len(prediction_indices)]

        # Now plot
        plt.figure(figsize=(30, 16))
        plt.plot(time_indices, self.true_values, label='Current Prices', linewidth=2, color='blue')
        plt.plot(prediction_indices, adjusted_predictions, label=f'Predictions {prediction_lag} days after', color='red', alpha=0.7, linewidth=2)

        # Labels, title, and legend
        plt.xlabel('Date', fontsize=16)
        plt.ylabel('Price', fontsize=16)
        plt.title('Actual vs Predicted Stock Prices', fontsize=18)
        plt.legend(loc='upper left', fontsize=14)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        max_index = len(time_indices) - 1

        for idx in range(prediction_lag, len(self.predictions) + prediction_lag, prediction_lag):
            if idx <= max_index:
                plt.axvline(x=time_indices[idx], color='gray', linestyle='--', alpha=0.5)

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def trading_strategy(self, window_size, starting_funds=50000):
        self.df.loc[self.df.index[window_size:], 'predictions'] = self.predictions
        funds = starting_funds
        stock_position = 0
        date = self.df.index[window_size:]
        stock_price = self.df['Close'].values[window_size:]

        
        actual_return = self.df['Return_5_days_later'].values[window_size:]
        actual_return_series = pd.Series(actual_return)
        predictions = self.df['predictions'].values[window_size:]

        portfolio_value = []
        buys, sells = [None] * len(predictions), [None] * len(predictions)
        for i in range(0, len(predictions)):
            current_return_window = actual_return_series[max(0, i-window_size):i]
            stats = current_return_window.describe()
            buy_threshold = stats["75%"]
            sell_threshold = stats["25%"] 
            
            past_predictions = predictions[i-10:i]
            buy_signals_count = sum(1 for p in past_predictions if p > buy_threshold)
            sell_signals_count = sum(1 for p in past_predictions if p < sell_threshold)

            if buy_signals_count >= 3:
                if funds > 0:
                    stocks_bought = funds // stock_price[i]
                    funds -= stocks_bought * stock_price[i]
                    stock_position += stocks_bought
                buys[i] = stock_price[i]
            elif sell_signals_count >= 3:
                if stock_position > 0:
                    funds += stock_position * stock_price[i]
                    stock_position = 0
                sells[i] = stock_price[i]

            portfolio_value.append(funds + stock_position * stock_price[i])

        # Calculate portfolio percentage growth
        portfolio_growth_percentage = [(value - starting_funds) / starting_funds * 100 for value in portfolio_value]

        # Visualization
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        ax1.set_ylabel('Stock Price', color='tab:blue')
        ln1 = ax1.plot(date, stock_price, color='tab:blue', alpha=0.6, label="Stock Price")
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Portfolio Value', color='tab:purple')
        ln2 = ax2.plot(date, portfolio_value, color='tab:purple', alpha=0.6, label="Portfolio Value")
        ax2.tick_params(axis='y', labelcolor='tab:purple')
        lns1 = ln1 + ln2
        labs1 = [l.get_label() for l in lns1]
        ax1.legend(lns1, labs1, loc='upper left')
        ax1.set_title("Stock Price and Portfolio Value")

        stock_growth_percentage = [(price - stock_price[0]) / stock_price[0] * 100 for price in stock_price]
        ln3 = ax3.plot(date, stock_growth_percentage, color='tab:green', alpha=0.6, label="Stock Growth %", linestyle='dashed')
        ax3.set_ylabel('Stock Growth (%)', color='tab:green')
        ax3.tick_params(axis='y', labelcolor='tab:green')
        ax4 = ax3.twinx()
        ax4.set_ylabel('Portfolio Growth (%)', color='tab:purple')
        ln4 = ax4.plot(date, portfolio_growth_percentage, color='tab:purple', alpha=0.6, label="Portfolio Growth %", linestyle='dotted')
        ax4.tick_params(axis='y', labelcolor='tab:purple')
        lns2 = ln3 + ln4
        labs2 = [l.get_label() for l in lns2]
        ax3.legend(lns2, labs2, loc='upper left')
        ax3.set_title("Stock Growth Percentage and Portfolio Growth Percentage")
        fig.tight_layout()
        plt.show()

        fig, ax1 = plt.subplots(figsize=(42, 21))

        # Stock Price with Buy/Sell actions
        ax1.set_ylabel('Stock Price', color='tab:blue')
        ax1.plot(date, stock_price, color='tab:blue', alpha=0.6, label="Stock Price")
        ax1.scatter(date, stock_price, color='black', marker='o', label="Points", alpha=0.2)
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Scatter plots for buy/sell signals
        buy_idx = [date[i] for i in range(len(buys)) if buys[i] is not None]
        sell_idx = [date[i] for i in range(len(sells)) if sells[i] is not None]
        buy_prices = [buys[i] for i in range(len(buys)) if buys[i] is not None]
        sell_prices = [sells[i] for i in range(len(sells)) if sells[i] is not None]

        ax1.scatter(buy_idx, buy_prices, color='g', label="Buy Signal", marker='^', alpha=1)
        ax1.scatter(sell_idx, sell_prices, color='r', label="Sell Signal", marker='v', alpha=1)

        # Adding legend and showing the plot
        ax1.legend(loc='upper left')
        ax1.set_title("Stock Price with Buy/Sell Actions")
        plt.show()
