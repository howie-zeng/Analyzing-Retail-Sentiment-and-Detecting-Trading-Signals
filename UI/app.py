from flask import Flask, render_template, request, jsonify
import pandas as pd
import os

app = Flask(__name__)

# List of stock files
stock_files = {
    'AAPL': 'UI/data/AAPL_2010-01-01_2023-01-01.csv',
    'AMC': 'UI/data/AMC_2010-01-01_2023-01-01.csv',
    'AMZN': 'UI/data/AMZN_2010-01-01_2023-01-01.csv',
    'BB': 'UI/data/BB_2010-01-01_2023-01-01.csv',
    'GME': 'UI/data/GME_2010-01-01_2023-01-01.csv',
    'GOOG': 'UI/data/GOOG_2010-01-01_2023-01-01.csv',
    'MSFT': 'UI/data/MSFT_2010-01-01_2023-01-01.csv',
    'PLTR': 'UI/data/PLTR_2010-01-01_2023-01-01.csv',
    'QQQ': 'UI/data/QQQ_2010-01-01_2023-01-01.csv',
    'RIVN': 'UI/data/RIVN_2010-01-01_2023-01-01.csv',
    'SOFI': 'UI/data/SOFI_2010-01-01_2023-01-01.csv',
    'SPY': 'UI/data/SPY_2010-01-01_2023-01-01.csv',
    'TSLA': 'UI/data/TSLA_2010-01-01_2023-01-01.csv'
}

# Function to load stock data
def load_stock_data(stock_name):
    file_path = stock_files.get(stock_name)
    if file_path:
        stock_data = pd.read_csv(file_path)
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        return stock_data
    return None

# Function to compute minimum and maximum prices for a stock
def compute_min_max_prices(stock_data):
    min_price = stock_data['Close'].min()
    max_price = stock_data['Close'].max()
    return min_price, max_price

@app.route('/')
def index():
    return render_template('index.html', stock_names=stock_files.keys())

@app.route('/filter', methods=['POST'])
def filter_data():
    stock_name = request.form.get('stock')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    
    stock_data = load_stock_data(stock_name)
    if stock_data is None:
        return jsonify({'error': 'Stock data not found'})
    
    # Extract and process form data
    min_price, max_price = compute_min_max_prices(stock_data)
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    min_price = float(request.form.get('min_price'))
    max_price = float(request.form.get('max_price'))
    min_volume = float(request.form.get('min_volume'))
    max_volume = float(request.form.get('max_volume'))
    chart_type = request.form.get('chart_type')

    # Filter the data based on the form input
    filtered_data = stock_data[
        (stock_data['Date'] >= pd.to_datetime(start_date)) & 
        (stock_data['Date'] <= pd.to_datetime(end_date)) &
        (stock_data['Close'] >= min_price) & 
        (stock_data['Close'] <= max_price) &
        (stock_data['Volume'] >= min_volume) & 
        (stock_data['Volume'] <= max_volume)
    ]

    # Construct the JSON response
    response_data = {
        'min_price': min_price,
        'max_price': max_price,
        'filtered_data': filtered_data.to_dict(orient='records')
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
