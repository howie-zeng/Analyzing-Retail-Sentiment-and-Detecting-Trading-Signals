import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


def calculate_ratio(dataframe, date_column, length):
    if not pd.api.types.is_datetime64_any_dtype(dataframe[date_column]):
        dataframe.loc[:, date_column] = pd.to_datetime(dataframe[date_column])

    max_date = dataframe[date_column].max()
    min_date = dataframe[date_column].min()
    date_difference = (max_date - min_date).days

    if date_difference == 0:
        return float('inf') 

    ratio = length / date_difference
    return ratio

def plot_daily_post_counts(df, stock_symbol):
    df['date'] = pd.to_datetime(df['date'])
    stock_df = df[df['stock'] == stock_symbol]
    daily_counts = stock_df.groupby(stock_df['date'].dt.date).size()

    mean_count = daily_counts.mean()
    percentile_75 = daily_counts.quantile(0.75)
    percentile_25 = daily_counts.quantile(0.25)

    sns.set(style="whitegrid") 
    plt.figure(figsize=(12, 8))
    plt.plot(daily_counts.index, daily_counts.values, marker='o', linestyle='-', color='b', label=stock_symbol)

    plt.axhline(y=mean_count, color='green', linestyle='--', label=f'Mean: {mean_count:.2f}')
    plt.axhline(y=percentile_75, color='red', linestyle='-.', label=f'75th Percentile: {percentile_75:.2f}')
    plt.axhline(y=percentile_25, color='red', linestyle='-.', label=f'25th Percentile: {percentile_25:.2f}')

    plt.title(f'Daily Posts about {stock_symbol}')
    plt.xlabel('Date')
    plt.ylabel('Number of Posts')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    plt.grid(True)

    max_count_date = daily_counts.idxmax()
    max_count = daily_counts.max()
    plt.annotate(f'Max: {max_count}', (max_count_date, max_count), textcoords="offset points", xytext=(0,10), ha='center')

    plt.tight_layout()
    plt.legend()
    plt.show()