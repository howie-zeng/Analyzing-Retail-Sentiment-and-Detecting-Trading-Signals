import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np

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
    
def compute_rsi_class(data):
    """Compute RSI class for the entire dataframe and add RSI thresholds."""
    thresholds = {
        'extremely_oversold': data['RSI'].quantile(0.05),
        'oversold': data['RSI'].quantile(0.10),
        'overbought': data['RSI'].quantile(0.90),
        'extremely_overbought': data['RSI'].quantile(0.95)
    }
    
    # Add thresholds to the dataframe
    data['rsi_extremely_oversold_threshold'] = thresholds['extremely_oversold']
    data['rsi_oversold_threshold'] = thresholds['oversold']
    data['rsi_overbought_threshold'] = thresholds['overbought']
    data['rsi_extremely_overbought_threshold'] = thresholds['extremely_overbought']
    
    # Compute the RSI class
    data['rsi_class'] = data['RSI'].apply(lambda x: rsi_class(x, thresholds))
    
    return data

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
    
    return data

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

