import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import ta
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, classification_report, confusion_matrix
)
import pywt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STOCKS = ["RIVN", "BB", "SOFI", "GME", "AMC", "PLTR", "TSLA", "AAPL", 'QQQ', "SPY", "DIA", "MSFT", "AMZN", "GOOG", '^IRX']
START_DATE = "2009-01-01"
END_DATE = "2023-01-01"
MAs = [5, 10, 20, 50, 100, 200]
PRICE_FEATURES_TO_CONVERT = ['MA5', 'MA10', 'MA20', 'MA50', 'MA100', 'MA200', 'BB_Upper', 'BB_Lower', 'Upper Band', 'SMA', 'Lower Band']
VOLUME_FEATURES_TO_CONVERT = ['Volume']
ORIGINAL_PRICE_FEATURES = ['Open', 'Close', 'High', 'Low', 'Adj Close']
RANDOM_STATE = 2023
rsi_palette = {
    'Extremely Oversold': 'red',
    'Oversold': 'orange',
    'Neutral': 'gray',
    'Overbought': 'lightgreen',
    'Extremely Overbought': 'green'
}

volume_palette = {
    'Volume Dip': 'red',
    'Minor Dip': 'orange',
    'Neutral': 'gray',
    'Minor Spike': 'lightgreen',
    'Volume Spike': 'green'
}


def rsi_class(rsi, thresholds):
    """Classify RSI into categories based on given thresholds."""
    if rsi <= thresholds['extremely_oversold']:
        return 'Extremely Oversold'
    elif rsi <= thresholds['oversold']:
        return 'Oversold'
    elif rsi >= thresholds['overbought']:
        return 'Overbought'
    elif rsi >= thresholds['extremely_overbought']:
        return 'Extremely Overbought'
    else:
        return 'Neutral'

def plot_rsi_category(data):
    """Plot RSI categories against stock price."""
    sns.set(rc={'figure.figsize':(28,8)})
    sns.set_style("whitegrid")
    plt.title("Examining RSI on movement of Price")
    ax = sns.scatterplot(x=data.index, y=data["Close"], hue=data["rsi_class"], palette=rsi_palette)
    plt.show()


def compute_rsi(data, window=14):
    """Compute RSI for given stock data."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def plot_stock_data(stock, data):
    """Plot stock data including price, volume, and RSI."""
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(28, 20), sharex=True)
    axes[0].plot(data['Date'], data['Close'], label='Close Price', color='blue')
    axes[0].set_title(f'{stock} Close Price and Trading Volume')
    axes[0].set_ylabel('Price')
    axes[0].legend(loc='upper left')

    ax2 = axes[0].twinx()
    ax2.bar(data['Date'], data['Volume'], label='Volume', color='red', alpha=0.7)
    ax2.set_ylabel('Volume')
    ax2.legend(loc='upper right')
    axes[1].plot(data['Date'], data['RSI'], label='RSI', color='orange')

    # Use dynamic thresholds for plotting the RSI lines
    for threshold_column, label in [('rsi_extremely_oversold_threshold', 'Extremely Oversold'), 
                                    ('rsi_oversold_threshold', 'Oversold'), 
                                    ('rsi_overbought_threshold', 'Overbought'), 
                                    ('rsi_extremely_overbought_threshold', 'Extremely Overbought')]:
        threshold_value = data[threshold_column].iloc[0]  # Assuming thresholds remain consistent within the data
        axes[1].axhline(threshold_value, linestyle='--', label=f'{label} ({threshold_value:.2f})')
    
    axes[1].set_title(f'{stock} RSI')
    axes[1].set_ylabel('RSI')
    axes[1].set_ylim(0, 100)
    axes[1].legend()
    
    for ax in [axes[0], ax2, axes[1]]:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)
    
    fig.suptitle(f'Analysis for {stock}', fontsize=16, y=1.03)
    plt.tight_layout()
    plt.show()
    
def plot_rsi_distribution(stock_data):
    """Plot the distribution of RSI given a dict of data."""
    sns.set_style("whitegrid")
    sns.set_palette("tab10")
    
    # KDE Plot
    plt.figure(figsize=(20, 7))
    for stock, data in stock_data.items():
        sns.kdeplot(data['RSI'], label=stock, fill=True, alpha=0.2)
    plt.title('RSI Distribution for All Stocks (Density Plot)', fontsize=16)
    plt.ylabel('Density', fontsize=14)
    plt.xlabel('RSI', fontsize=14)
    plt.legend(title='Stock', fontsize=12)
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.show()
    
    # Prepare data for Box Plot
    all_data = []
    for stock, data in stock_data.items():
        temp_df = data[['RSI']].copy()
        temp_df['Stock'] = stock
        all_data.append(temp_df)
    
    rsi_data_long = pd.concat(all_data, axis=0).reset_index(drop=True)  # Reset the index
    
    # Box Plot with Stripplot overlay
    plt.figure(figsize=(20, 7))
    sns.boxplot(x="Stock", y="RSI", data=rsi_data_long, palette="tab10", boxprops=dict(alpha=.7), width=0.6)
    sns.stripplot(x="Stock", y="RSI", data=rsi_data_long, color=".25", size=3, jitter=True)
    plt.title('RSI Distribution for All Stocks (Box Plot with Stripplot)', fontsize=16)
    plt.ylabel('RSI', fontsize=14)
    plt.xlabel('Stock', fontsize=14)
    plt.tight_layout()
    plt.show()


def calculate_moving_average(data, window_size):
    return data.rolling(window=window_size).mean()


def volume_class(row):
    """Classify Volume into categories based on relative difference and thresholds."""
    rel_diff = row['Relative_Difference']
    thresholds = row[['volume_dip', 'minor_dip', 'minor_spike', 'volume_spike']]
    
    if rel_diff <= thresholds['volume_dip']:
        return 'Volume Dip'
    elif rel_diff <= thresholds['minor_dip']:
        return 'Minor Dip'
    elif rel_diff >= thresholds['volume_spike']:
        return 'Volume Spike'
    elif rel_diff >= thresholds['minor_spike']:
        return 'Minor Spike'
    else:
        return 'Neutral'

def plot_volume_distribution(stock_data):
    """Plot the distribution of relative difference given a dict of data."""
    sns.set_style("whitegrid")
    plt.figure(figsize=(28, 14))
    for stock, data in stock_data.items():
        sns.kdeplot(data['Relative_Difference'], label=stock, fill=True)  
    plt.title('Relative Difference Distribution for All Stocks')
    plt.xlabel('Relative Difference')
    plt.ylabel('Density')
    plt.legend(title='Stock')
    plt.show()

def plot_volume_category(data):
    """Plot Volume categories against stock price."""
    sns.set(rc={'figure.figsize':(28,8)})
    sns.set_style("whitegrid")
    plt.title("Examining Volume on movement of Price")
    ax = sns.scatterplot(x=data.index, y=data["Close"], hue=data["volume_class"], palette=volume_palette)
    plt.show()


def categorize_movement(return_val):
    if -0.002 <= return_val <= 0.002:
        return 'Stable'
    elif 0.002 < return_val <= 0.01:
        return 'Slight Uptrend'
    elif -0.01 <= return_val < -0.002:
        return 'Slight Downtrend'
    elif 0.01 < return_val <= 0.02:
        return 'Moderate Uptrend'
    elif -0.02 <= return_val < -0.01:
        return 'Moderate Downtrend'
    elif return_val > 0.02:
        return 'Strong Uptrend'
    else:
        return 'Strong Downtrend'
    
    
def load_stock_data(stock, current_path):
    file_pattern = f"{current_path}/data/{stock}*.csv"
    csv_files = glob.glob(file_pattern)
    
    if not csv_files:
        print(f"No CSV file found for {stock}")
        return None 

    file_path = csv_files[0]
    data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    
    return data

def plot_predictions(df, predictions, window_size, stock):
    plt.figure(figsize=(40, 20))
    
    # Extract necessary data
    dates = df['Date'][window_size:].tolist()
    actual_movements = df['movement_category'][window_size:].tolist()
    close_prices = df['Close'][window_size:].tolist()
    
    # Determine correct and incorrect predictions
    correct_dates = [dates[i] for i, (actual, pred) in enumerate(zip(actual_movements, predictions)) if actual == pred]
    incorrect_dates = [dates[i] for i, (actual, pred) in enumerate(zip(actual_movements, predictions)) if actual != pred]
    correct_prices = [close_prices[i] for i, (actual, pred) in enumerate(zip(actual_movements, predictions)) if actual == pred]
    incorrect_prices = [close_prices[i] for i, (actual, pred) in enumerate(zip(actual_movements, predictions)) if actual != pred]
    
    # Plotting the actual close prices
    plt.plot(dates, close_prices, label='Actual Close Price', color='blue', linewidth=2.5)
    
    # Plotting the points where predictions were correct
    plt.scatter(correct_dates, correct_prices, color='green', label='Correct Prediction', s=50, alpha=0.7)
    
    # Plotting the points where predictions were incorrect
    plt.scatter(incorrect_dates, incorrect_prices, color='red', label='Incorrect Prediction', s=150, marker='x', alpha=0.7)
    
    # Adjust x-axis to only show the first day of each month for clarity
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1)) # Display first day of each month
    
    # Plot settings
    plt.title(f'{stock} Stock Price with Movement Predictions', fontsize=24)
    plt.xlabel('Date', fontsize=20)
    plt.ylabel('Close Price', fontsize=20)
    plt.xticks(rotation=45, fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=18, loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(true_values, predictions, labels_order):
    """
    Plots a confusion matrix for the given true values and predictions using the specified labels order.
    
    Args:
    - true_values (list): List of true labels.
    - predictions (list): List of predicted labels.
    - labels_order (list): List of label names in the desired order.
    """
    
    cm = confusion_matrix(true_values, predictions, labels=labels_order)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                xticklabels=labels_order, 
                yticklabels=labels_order)
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('True', fontsize=16)
    plt.title('Confusion Matrix', fontsize=20)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()
    accuracy = accuracy_score(true_values, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(true_values, predictions))

def post_process(df, predictions, window_size):
    df["predicted_close"] = df["Close"]
    df.loc[df.index[window_size:], "predicted_close"] = predictions
    df["predicted_return"] = df['predicted_close'].pct_change().fillna(0)
    df["predicted_movement_category"] = df['predicted_return'].apply(categorize_movement)
    return df

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_returns(prices):
    prices = np.array(prices)
    returns = np.diff(prices) / prices[:-1] * 100
    return returns

def calculate_returns(prices):
    prices = np.array(prices)
    returns = np.diff(prices) / prices[:-1] * 100
    return returns

def calculate_return_difference(y_price, y_predicted_price):
    actual_returns = calculate_returns(y_price)
    predicted_returns = calculate_returns(y_predicted_price)
    min_length = min(len(actual_returns), len(predicted_returns))
    return_difference = actual_returns[:min_length] - predicted_returns[:min_length]
    
    return return_difference

### Indicators
def calculate_rsi(data, window=14):
    # Missing RSI calculation
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_rsi_class(data):
    rsi = calculate_rsi(data)
    conditions = [
        (rsi >= 70),
        (rsi <= 30),
        (rsi.between(30, 70))
    ]
    choices = [1, -1, 0]
    return pd.Series(pd.cut(rsi, bins=[0, 30, 70, 100], labels=[-1, 0, 1], right=True))


def compute_volume_class(data, window=10):
    """Compute Volume class based on relative difference and add to dataframe."""
    
    # Compute the 10-day moving average for volume
    data['MA_Volume'] = data['Volume'].rolling(window=window).mean()
    
    # Compute the relative difference
    data['Relative_Difference'] = (data['Volume'] - data['MA_Volume']) / data['MA_Volume']
    
    # Define thresholds
    thresholds = {
        'volume_dip': data['Relative_Difference'].quantile(0.10),
        'minor_dip': data['Relative_Difference'].quantile(0.30),
        'minor_spike': data['Relative_Difference'].quantile(0.70),
        'volume_spike': data['Relative_Difference'].quantile(0.90)
    }
    for key, value in thresholds.items():
        data[key] = value
    
    # Compute the Volume class
    data['volume_class'] = data.apply(volume_class, axis=1)
    
    return data['volume_class']

def calculate_mas(data, periods, column_name="Close"):
    """
    Calculate moving averages for specified periods.

    Parameters:
    - data: pandas DataFrame containing stock data.
    - periods: list of integers specifying the moving average periods. Default is [5, 10, 20, 50, 200].
    - column_name: name of the column in the DataFrame to compute the MAs for. Default is "Close".

    Returns:
    - pandas DataFrame with added MA columns.
    """
    
    for period in periods:
        ma_label = f"MA{period}"
        data[ma_label] = data[column_name].rolling(window=period).mean()
    
    return data

def calculate_wvad(data, period=14):
    wvad_numerator = (data['Close'] - data['Low']) - (data['High'] - data['Close'])
    wvad_denominator = data['High'] - data['Low']
    wvad = wvad_numerator / wvad_denominator * data['Volume']
    return wvad.rolling(window=period).sum()

def calculate_macd(stock_data, short_window=12, long_window=26, signal_window=9):
    short_ema = stock_data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = stock_data['Close'].ewm(span=long_window, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    return macd_line - signal_line, macd_line, signal_line


def calculate_roc(stock_data, period=14):
    """
    Calculate the Rate of Change (ROC) for a given stock DataFrame.

    Parameters:
        stock_data (pd.DataFrame): DataFrame containing stock data with a column 'Close'.
        period (int): The period for calculating ROC. Default is 14.

    Returns:
        pd.Series: Series containing ROC values for each day within the specified period.
    """
    # Calculate ROC
    roc = stock_data['Close'].pct_change(periods=period) * 100

    return roc

def calculate_cci(data, period=20):
    TP = (data['High'] + data['Low'] + data['Close']) / 3
    SMA_TP = TP.rolling(window=period).mean()
    MD = np.abs(TP - SMA_TP).rolling(window=period).mean()
    CCI = (TP - SMA_TP) / (0.015 * MD)
    return CCI

def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    sma = data['Close'].rolling(window=window).mean()
    std_dev = data['Close'].rolling(window=window).std()
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return upper_band, lower_band, sma


def calculate_smi(data, period=14, signal_period=3):
    lowest_low = data['Close'].rolling(window=period).min()
    highest_high = data['Close'].rolling(window=period).max()
    SO = ((data['Close'] - lowest_low) / (highest_high - lowest_low)) * 100
    EMA_SO = SO.ewm(span=signal_period, adjust=False).mean()
    SMI = SO - EMA_SO
    return SMI


def calculate_atr(data, period=14):
    high_low = data['High'] - data['Low']
    high_prevclose = abs(data['High'] - data['Close'].shift(1))
    low_prevclose = abs(data['Low'] - data['Close'].shift(1))
    TR = pd.concat([high_low, high_prevclose, low_prevclose], axis=1).max(axis=1)
    ATR = TR.ewm(span=period, adjust=False).mean()
    return ATR

def cm_williams_vix_fix(close_prices, low_prices, pd_=22, bbl=20, mult=2.0, lb=50, ph=0.85, pl=1.01):
    """
    Compute the CM Williams Vix Fix values.

    Parameters:
    - close_prices: Series of close prices.
    - low_prices: Series of low prices.
    - pd_, bbl, mult, lb, ph, pl: Various parameters for calculation.

    Returns:
    - A DataFrame with 'wvf', 'upperBand', 'rangeHigh', and 'color' columns.
    """
    
    # Calculate the highest close over the pd_ period
    highest_close = close_prices.rolling(window=pd_).max()
    
    # Williams Vix Fix calculation
    wvf = ((highest_close - low_prices) / highest_close) * 100
    
    # Calculate the midLine using a simple moving average
    midLine = wvf.rolling(window=bbl).mean()
    
    # Calculate standard deviation over the bbl period
    sDev = mult * wvf.rolling(window=bbl).std()
    
    # Calculate the range high value
    rangeHigh = wvf.rolling(window=lb).max() * ph
    
    # Determine color based on conditions
    color = [1 if w >= midLine.iloc[i] + sDev.iloc[i] or w >= rangeHigh.iloc[i] else 0 for i, w in enumerate(wvf)] # 1 for line and 0 for grey

    return pd.DataFrame({'WVF': wvf, 'upperBand': midLine + sDev, 'rangeHigh': rangeHigh, 'WVF_color': color})


def bollinger_rsi_strategy(close_prices: pd.Series) -> pd.DataFrame:
    data = pd.DataFrame(close_prices, columns=['Close'])
    
    # Parameters
    RSI_length = 6
    RSI_overSold = 50
    RSI_overBought = 50
    BB_length = 200
    BB_mult = 2

    # Calculate RSI
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=RSI_length).rsi()

    # Calculate Bollinger Bands
    bb = ta.volatility.BollingerBands(data['Close'], window=BB_length, window_dev=BB_mult)
    data['BB_Upper'] = bb.bollinger_hband()
    data['BB_Lower'] = bb.bollinger_lband()
    # Buy and sell signals
    data['Buy_Signal'] = (data['RSI'].shift(1) < RSI_overSold) & (data['RSI'] > RSI_overSold) & (data['Close'].shift(1) < data['BB_Lower']) & (data['Close'] > data['BB_Lower'])

    data['Sell_Signal'] = (data['RSI'].shift(1) > RSI_overBought) & (data['RSI'] < RSI_overBought) & (data['Close'].shift(1) > data['BB_Upper']) & (data['Close'] < data['BB_Upper'])

    return data[['Buy_Signal', 'Sell_Signal', 'BB_Upper', 'BB_Lower']]

def on_balance_volume(data):
    """
    Calculate On-Balance Volume (OBV) and add it to the DataFrame.
    """
    obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
    data['OBV'] = obv
    return data

def volume_price_trend(data):
    """
    Calculate Volume Price Trend (VPT) and add it to the DataFrame.
    """
    vpt = (data['Volume'] * (data['Close'].diff() / data['Close'])).cumsum()
    data['VPT'] = vpt
    return data

def money_flow_index(data, period=14):
    """
    Calculate Money Flow Index (MFI) and add it to the DataFrame.
    """
    typical_price = (data[['High', 'Low', 'Close']].sum(axis=1)) / 3
    money_flow = typical_price * data['Volume']
    
    pos_flow = pd.Series(np.where(typical_price > typical_price.shift(), money_flow, 0), index=data.index)
    neg_flow = pd.Series(np.where(typical_price < typical_price.shift(), money_flow, 0), index=data.index)
    
    money_ratio = pos_flow.rolling(period).sum() / neg_flow.rolling(period).sum()
    
    data['MFI'] = 100 - (100 / (1 + money_ratio))
    return data


def accumulation_distribution(data):
    """
    Calculate Accumulation/Distribution Line (A/D) and add it to the DataFrame.
    """
    high_minus_low = data['High'] - data['Low']
    close_minus_low = data['Close'] - data['Low']
    high_minus_close = data['High'] - data['Close']

    # Avoid division by zero by replacing zeros with NaNs
    with np.errstate(divide='ignore', invalid='ignore'):
        clv = (close_minus_low - high_minus_close) / high_minus_low
        clv[np.isinf(clv)] = 0  # Handling cases where high is equal to low
        clv.fillna(0, inplace=True)

    data['AD'] = (clv * data['Volume']).cumsum()
    return data

import pandas as pd
import pywt

def apply_stationary_wavelet_transform(data, wavelet='haar', level=1):
    coefficient_dfs = []  
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]) and col != 'Close':
            coeffs = pywt.swt(data[col], wavelet, level=level, trim_approx=False)
            # Flatten the coefficients for each level and column
            flattened_coeffs = [item for sublist in coeffs for item in sublist]
            coeff_df = pd.DataFrame(flattened_coeffs).T
            coeff_df.columns = [f'{col}_level_{i}_coeff_{j}' for i in range(len(coeffs)) for j in range(2)]
            coefficient_dfs.append(coeff_df)
        else:
            if col != 'Close':
                print(f'{col} is not numerical or is the target variable')

    if coefficient_dfs:
        df = pd.concat(coefficient_dfs, axis=1)
    else:
        df = pd.DataFrame() 
    return df


def reconstruct_from_coeffs(coeffs, wavelet='haar'):
    reconstructed_data = pywt.waverec(coeffs, wavelet)
    return reconstructed_data

def preprocess_stock_data(stock_data, STOCKS):
    processed_data = {}
    MAs = [5, 10, 20, 50, 100, 200]
    for stock in STOCKS:
        data = stock_data[stock].copy()
        if len(data) % 2 != 0:
            data = data[:-1]
        data['Date'] = pd.to_datetime(data['Date'])
        if stock != "^IRX":
            data['RSI'] = compute_rsi(data['Close'])
            data['rsi_class'] = compute_rsi_class(data)
            data = calculate_mas(data, MAs, column_name="Close")
            data['WVAD'] = calculate_wvad(data, period=14)
            data['ROC'] = calculate_roc(data, period=14)
            data['MACD'], data['macd_line'], data['signal_line'] = calculate_macd(data, short_window=12, long_window=26, signal_window=9)
            data['CCI'] = calculate_cci(data, period=20)
            data['Upper Band'], data['Lower Band'], data['SMA'] = calculate_bollinger_bands(data, window=20, num_std_dev=2)
            data['SMI'] = calculate_smi(data, period=14, signal_period=3)
            data['ATR'] = calculate_atr(data, period=14)
            data[['WVF', 'upperBand', 'rangeHigh', 'WVF_color']] = cm_williams_vix_fix(data['Close'], data['Low'])
            data[['Buy_Signal', 'Sell_Signal', 'BB_Upper', 'BB_Lower']] = bollinger_rsi_strategy(data['Close'])
            data = on_balance_volume(data)
            data = volume_price_trend(data)
            data = money_flow_index(data)
            data = accumulation_distribution(data)
            data = data.dropna()
        processed_data[stock] = data
        print(f"Data fetched for {stock}")
    return processed_data

def create_sequences(input_data, target_data, sequence_length, prediction_length):
    xs, ys = [], []
    for i in range(len(input_data) - sequence_length - prediction_length + 1):
        xs.append(input_data[i:(i + sequence_length)])
        ys.append(target_data[i + sequence_length + prediction_length - 1])  # Only the price on the 5th day
    return np.array(xs), np.array(ys)


def prepare_data_and_model(df_stock, sequence_length, prediction_length, test_size=0.2):
    if len(df_stock) % 2 != 0:  # must be even
        df_stock = df_stock[:-1]
    
    df_stock_swt = apply_stationary_wavelet_transform(df_stock)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_stock_swt = scaler.fit_transform(df_stock_swt)

    close_prices_Y = df_stock['Close']
    
    X, y = create_sequences(df_stock_swt, close_prices_Y, sequence_length, prediction_length)
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    # y = scaler_y.fit_transform(y)
    # scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    train_size = int((1 - test_size) * len(X_tensor))
    X_train_tensor = X_tensor[:train_size]
    y_train_tensor = y_tensor[:train_size]
    X_test_tensor = X_tensor[train_size:]
    y_test_tensor = y_tensor[train_size:]
    
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    test_data = TensorDataset(X_test_tensor, y_test_tensor)
    
    batch_size = 32
    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    
    input_size = X.shape[2]
    
    return train_loader, test_loader, input_size, scaler_y




