import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output


# List of stock data file paths and corresponding stock names
stock_files = [
    {'file_path': 'data/AAPL_2010-01-01_2023-01-01.csv', 'stock_name': 'AAPL'},
    {'file_path': 'data/AMC_2010-01-01_2023-01-01.csv', 'stock_name': 'AMC'},
    {'file_path': 'data/AMZN_2010-01-01_2023-01-01.csv', 'stock_name': 'AMZN'},
    {'file_path': 'data/BB_2010-01-01_2023-01-01.csv', 'stock_name': 'BB'},
    {'file_path': 'data/GME_2010-01-01_2023-01-01.csv', 'stock_name': 'GME'},
    {'file_path': 'data/GOOG_2010-01-01_2023-01-01.csv', 'stock_name': 'GOOG'},
    {'file_path': 'data/MSFT_2010-01-01_2023-01-01.csv', 'stock_name': 'MSFT'},
    {'file_path': 'data/PLTR_2010-01-01_2023-01-01.csv', 'stock_name': 'PLTR'},
    {'file_path': 'data/QQQ_2010-01-01_2023-01-01.csv', 'stock_name': 'QQQ'},
    {'file_path': 'data/RIVN_2010-01-01_2023-01-01.csv', 'stock_name': 'RIVN'},
    {'file_path': 'data/SOFI_2010-01-01_2023-01-01.csv', 'stock_name': 'SOFI'},
    {'file_path': 'data/SPY_2010-01-01_2023-01-01.csv', 'stock_name': 'SPY'},
    {'file_path': 'data/TSLA_2010-01-01_2023-01-01.csv', 'stock_name': 'TSLA'}
]

# Load the selected stock data
selected_stock_data = None

# UI Components
stock_dropdown = widgets.Dropdown(
    options=[(stock['stock_name'], i) for i, stock in enumerate(stock_files)],
    description='Select Stock:',
    disabled=False,
)

start_date_picker = widgets.DatePicker(
    description='Start Date',
    disabled=False
)

end_date_picker = widgets.DatePicker(
    description='End Date',
    disabled=False
)

price_range_slider = widgets.FloatRangeSlider(
    min=0,
    max=1,
    step=0.01,
    description='Price Range:',
    continuous_update=False
)

volume_range_slider = widgets.FloatRangeSlider(
    min=0,
    max=1,
    step=0.01,
    description='Volume Range:',
    continuous_update=False
)

chart_type_dropdown = widgets.Dropdown(
    options=['Line Chart', 'Candlestick Chart'],
    value='Line Chart',
    description='Chart Type:',
    disabled=False,
)

filter_button = widgets.Button(description='Apply Filters')
output = widgets.Output()

# Function to load selected stock data
def load_selected_stock_data(change):
    global selected_stock_data
    stock_index = change.new
    if stock_index is not None:
        file_path = stock_files[stock_index]['file_path']
        selected_stock_data = pd.read_csv(file_path)
        selected_stock_data['Date'] = pd.to_datetime(selected_stock_data['Date'])
        clear_output()
        print(f"Loaded {stock_files[stock_index]['stock_name']} stock data.")

stock_dropdown.observe(load_selected_stock_data, names='value')

# Function to Update Output
def filter_data(b):
    with output:
        clear_output()
        if selected_stock_data is None:
            print("Please select a stock.")
            return

        filtered_data = selected_stock_data[
            (selected_stock_data['Date'] >= pd.to_datetime(start_date_picker.value)) & 
            (selected_stock_data['Date'] <= pd.to_datetime(end_date_picker.value)) &
            (selected_stock_data['Close'] >= price_range_slider.value[0]) & 
            (selected_stock_data['Close'] <= price_range_slider.value[1]) &
            (selected_stock_data['Volume'] >= volume_range_slider.value[0]) & 
            (selected_stock_data['Volume'] <= volume_range_slider.value[1])
        ]

        if filtered_data.empty:
            print("No data to display for the selected filters.")
            return

        if chart_type_dropdown.value == 'Line Chart':
            # Plot the stock price as a line chart
            plt.figure(figsize=(10, 4))
            plt.plot(filtered_data['Date'], filtered_data['Close'], label='Close Price')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title(f"{stock_files[stock_dropdown.value]['stock_name']} Stock Price")
            plt.legend()
            plt.show()
        elif chart_type_dropdown.value == 'Candlestick Chart':
            # Create a candlestick chart
            fig = go.Figure(data=[go.Candlestick(
                x=filtered_data['Date'],
                open=filtered_data['Open'],
                high=filtered_data['High'],
                low=filtered_data['Low'],
                close=filtered_data['Close']
            )])

            fig.update_layout(
                title=f"{stock_files[stock_dropdown.value]['stock_name']} Candlestick Chart",
                xaxis_title='Date',
                yaxis_title='Price',
            )

            # Display the candlestick chart
            fig.show()

# Link the filter button to the filter_data function
filter_button.on_click(filter_data)

# Display UI components
display(stock_dropdown, start_date_picker, end_date_picker, price_range_slider, volume_range_slider, chart_type_dropdown, filter_button, output)
